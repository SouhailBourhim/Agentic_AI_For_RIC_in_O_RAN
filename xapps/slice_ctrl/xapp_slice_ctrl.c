/*
 * Slice-control xApp (M07b): turns "PRBs per slice" lines on stdin into E2 SLICE control messages.
 *
 * Adapted from FlexRIC v3.0.0 examples/xApp/c/slice/xapp_slice_moni_ctrl.c (SPDX:
 * LicenseRef-CSSL-1.0): same message (SLICE service model, ADD with STATIC slices), same call
 * (control_sm_xapp_api). What changed:
 *   - no SLICE subscription: the OAI gNB's SLICE reports are random test data, and the KPIs come
 *     from the KPM xApp (M07a);
 *   - long-lived: one control per stdin line, until stdin closes. The caller is
 *     backends/flexric.py, inside the RL xApp container. This program decides nothing:
 *       "64 32 10"  -> eMBB PRBs [0,63], URLLC [64,95], BE [96,105]  (ADD, three STATIC ranges)
 *       "del"       -> DEL: the gNB goes back to its stock scheduler over the whole cell
 *   - one JSON line per command on stdout: what was sent and how long the acknowledge took.
 *
 * The FlexRIC xApp API does not hand the gNB's answer (OK or refused + reason) back to the caller:
 * control_sm_xapp_api returns success as soon as a RIC Control Acknowledge arrives. The answer is
 * in the acknowledge on the wire and in the gNB log ("SLICE CONTROL OK" / "REFUSED"). If no
 * acknowledge comes back in ~6 s, the FlexRIC library aborts the process; the caller notices.
 *
 * Slice ids 0, 1, 2 = eMBB, URLLC, BE, as in the gNB patch (testbed/gnb/slice-control.patch);
 * tests/test_testbed_config.py checks both tables match.
 */

#include "../../../../src/xApp/e42_xapp_api.h"
#include "../../../../src/util/alg_ds/alg/defer.h"
#include "../../../../src/util/time_now_us.h"
#include "../../../../src/sm/slice_sm/slice_sm_id.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// SLICES: (id, label) in the order of the stdin line; same ids as the gNB patch
static const char* const labels[] = {"embb", "urllc", "be"};
#define N_SLICES (sizeof(labels) / sizeof(labels[0]))

static char* copy_str(const char* s, uint32_t* len)
{
  *len = strlen(s);
  char* out = malloc(*len);
  assert(out != NULL);
  memcpy(out, s, *len);
  return out;
}

/* ADD: slice i gets PRBs [start_i, start_i + prbs[i] - 1], laid out one after the other. */
static slice_ctrl_req_data_t add_request(const uint32_t prbs[N_SLICES])
{
  slice_ctrl_req_data_t req = {0};
  req.msg.type = SLICE_CTRL_SM_V0_ADD;
  ul_dl_slice_conf_t* dl = &req.msg.u.add_mod_slice.dl;
  dl->sched_name = copy_str("PF", &dl->len_sched_name);
  dl->len_slices = N_SLICES;
  dl->slices = calloc(N_SLICES, sizeof(fr_slice_t));
  assert(dl->slices != NULL);
  uint32_t start = 0;
  for (uint32_t i = 0; i < N_SLICES; i++) {
    fr_slice_t* s = &dl->slices[i];
    s->id = i;
    s->label = copy_str(labels[i], &s->len_label);
    s->sched = copy_str("PF", &s->len_sched);
    s->params.type = SLICE_ALG_SM_V0_STATIC;
    s->params.u.sta.pos_low = start;
    s->params.u.sta.pos_high = start + prbs[i] - 1;  // prbs[i] == 0 wraps; the gNB refuses it
    start += prbs[i];
  }
  ul_dl_slice_conf_t* ul = &req.msg.u.add_mod_slice.ul;  // uplink untouched: no slices
  ul->sched_name = copy_str("PF", &ul->len_sched_name);
  return req;
}

static slice_ctrl_req_data_t del_request(void)
{
  slice_ctrl_req_data_t req = {0};
  req.msg.type = SLICE_CTRL_SM_V0_DEL;
  del_slice_conf_t* del = &req.msg.u.del_slice;
  del->len_dl = N_SLICES;
  del->dl = calloc(N_SLICES, sizeof(uint32_t));
  assert(del->dl != NULL);
  for (uint32_t i = 0; i < N_SLICES; i++)
    del->dl[i] = i;
  return req;
}

int main(int argc, char* argv[])
{
  setvbuf(stdout, NULL, _IOLBF, 0);  // one JSON line at a time to the caller
  fr_args_t args = init_fr_args(argc, argv);
  init_xapp_api(&args);
  sleep(1);

  e2_node_arr_xapp_t nodes = e2_nodes_xapp_api();
  defer({ free_e2_node_arr_xapp(&nodes); });
  if (nodes.len == 0) {
    printf("{\"event\":\"no_e2_node\"}\n");
    return 1;
  }
  global_e2_node_id_t* node = &nodes.n[0].id;  // the testbed has one gNB
  printf("{\"event\":\"ready\"}\n");

  char line[128];
  while (fgets(line, sizeof(line), stdin) != NULL) {
    uint32_t prbs[N_SLICES] = {0};
    slice_ctrl_req_data_t req;
    if (strncmp(line, "del", 3) == 0) {
      req = del_request();
    } else if (sscanf(line, "%u %u %u", &prbs[0], &prbs[1], &prbs[2]) == 3) {
      req = add_request(prbs);
    } else {
      line[strcspn(line, "\n")] = '\0';
      printf("{\"event\":\"bad_line\",\"line\":\"%.40s\"}\n", line);
      continue;
    }
    int64_t const t0 = time_now_us();
    control_sm_xapp_api(node, SM_SLICE_ID, &req);  // blocks until the RIC Control Acknowledge
    int64_t const t1 = time_now_us();
    if (req.msg.type == SLICE_CTRL_SM_V0_DEL)
      printf("{\"event\":\"ack\",\"cmd\":\"del\",\"sent_us\":%ld,\"rtt_ms\":%.3f}\n", t0, (t1 - t0) / 1000.0);
    else
      printf("{\"event\":\"ack\",\"cmd\":\"add\",\"prbs\":[%u,%u,%u],\"sent_us\":%ld,\"rtt_ms\":%.3f}\n", prbs[0],
             prbs[1], prbs[2], t0, (t1 - t0) / 1000.0);
    free_slice_ctrl_msg(&req.msg);
  }

  while (try_stop_xapp_api() == false)
    usleep(1000);
  printf("{\"event\":\"stopped\"}\n");
  return 0;
}
