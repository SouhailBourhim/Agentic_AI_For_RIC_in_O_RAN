/*
 * KPM monitoring xApp for the three testbed slices (M07a).
 *
 * Adapted from FlexRIC v3.0.0 examples/xApp/c/monitor/xapp_kpm_moni.c (SPDX: LicenseRef-CSSL-1.0):
 * same connection, subscription and callback structure. What changed:
 *   - one KPM REPORT style 4 subscription PER SLICE, whose matching condition is that slice's
 *     S-NSSAI as 4 bytes (SST + 3-byte SD). The stock xApp sends 1 byte (SST 1 only), which the
 *     OAI gNB matches only against UEs whose SD is 0xFFFFFF (ran_func_kpm.c, nssai_matches);
 *   - only three per-UE measurements: DRB.RlcSduDelayDl (first, see meas_names), DRB.UEThpDl,
 *     RRU.PrbTotDl;
 *   - one callback per slice, because the FlexRIC callback receives no subscription id;
 *   - output: one JSON line per UE per report on stdout. All the mapping to the project's KPI
 *     record is done in Python (xapps/kpm_monitor.py), where it is tested.
 * Style 1 (cell-level), distribution labels and the generic printing are removed.
 *
 * The slice table must match the testbed configs (tests/test_testbed_config.py checks it).
 */

#include "../../../../src/xApp/e42_xapp_api.h"
#include "../../../../src/util/alg_ds/alg/defer.h"
#include "../../../../src/util/time_now_us.h"
#include "../../../../src/util/e.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static uint64_t const period_ms = 1000;  // report period and granularity period

typedef struct {
  const char* name;
  uint8_t sst;
  uint32_t sd;
} slice_t;

// SLICES: (name, SST, SD) as in testbed/oai-cn5g/conf/config.yaml and rfsim-flexric/conf/gnb.conf
static const slice_t slices[] = {
    {"embb", 1, 0x000001},
    {"urllc", 2, 0x000001},
    {"be", 1, 0x000002},
};
#define N_SLICES (sizeof(slices) / sizeof(slices[0]))

// DRB.RlcSduDelayDl must come first: every RLC measurement the gNB fills reads the RLC statistics
// and then RESETS the delay average (OAI get_rlc_stats_per_drb -> nr_rlc_activate_avg_time_to_tx),
// so a delay read right after DRB.UEThpDl always sees an empty average, i.e. 0.
static const char* const meas_names[] = {"DRB.RlcSduDelayDl", "DRB.UEThpDl", "RRU.PrbTotDl"};
#define N_MEAS (sizeof(meas_names) / sizeof(meas_names[0]))

static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;  // callbacks run on the xApp's threads

/* Print one value as JSON: a number, or null when the gNB sent no value. */
static void print_value(meas_record_lst_t const* rec)
{
  if (rec->value == INTEGER_MEAS_VALUE)
    printf("%u", rec->int_val);
  else if (rec->value == REAL_MEAS_VALUE)
    printf("%.3f", rec->real_val);
  else
    printf("null");
}

/* One line per UE per granularity period:
 * {"slice":"urllc","collect_start_us":...,"recv_us":...,"amf_ue_ngap_id":...,"DRB.UEThpDl":...,...} */
static void print_ue(const char* slice, uint64_t collect_start_us, int64_t recv_us,
                     ue_id_e2sm_t const* ue, kpm_ind_msg_format_1_t const* frm_1)
{
  for (size_t d = 0; d < frm_1->meas_data_lst_len; d++) {
    meas_data_lst_t const* data = &frm_1->meas_data_lst[d];
    printf("{\"slice\":\"%s\",\"collect_start_us\":%lu,\"recv_us\":%ld", slice, collect_start_us, recv_us);
    if (ue->type == GNB_UE_ID_E2SM)
      printf(",\"amf_ue_ngap_id\":%lu", ue->gnb.amf_ue_ngap_id);
    // One label per measurement (noLabel), so record i belongs to measurement i.
    for (size_t i = 0; i < frm_1->meas_info_lst_len && i < data->meas_record_len; i++) {
      char* name = cp_ba_to_str(frm_1->meas_info_lst[i].meas_type.name);
      printf(",\"%s\":", name);
      print_value(&data->meas_record_lst[i]);
      free(name);
    }
    if (data->incomplete_flag != NULL && *data->incomplete_flag == TRUE_ENUM_VALUE)
      printf(",\"incomplete\":true");
    printf("}\n");
  }
}

static void on_report(size_t slice_idx, sm_ag_if_rd_t const* rd)
{
  assert(rd != NULL && rd->type == INDICATION_MSG_AGENT_IF_ANS_V0 && rd->ind.type == KPM_STATS_V3_0);
  kpm_ind_data_t const* ind = &rd->ind.kpm.ind;
  uint64_t const start_us = ind->hdr.kpm_ric_ind_hdr_format_1.collectStartTime;
  int64_t const now_us = time_now_us();
  const char* slice = slices[slice_idx].name;

  pthread_mutex_lock(&mtx);
  if (ind->msg.type == FORMAT_3_INDICATION_MESSAGE) {
    kpm_ind_msg_format_3_t const* msg = &ind->msg.frm_3;
    for (size_t u = 0; u < msg->ue_meas_report_lst_len; u++)
      print_ue(slice, start_us, now_us, &msg->meas_report_per_ue[u].ue_meas_report_lst,
               &msg->meas_report_per_ue[u].ind_msg_format_1);
  } else {
    printf("{\"event\":\"unexpected_format\",\"slice\":\"%s\",\"format\":%d}\n", slice, ind->msg.type);
  }
  pthread_mutex_unlock(&mtx);
}

// The callback type is void(*)(sm_ag_if_rd_t const*): no user data, so one small function per slice.
static void cb_embb(sm_ag_if_rd_t const* rd) { on_report(0, rd); }
static void cb_urllc(sm_ag_if_rd_t const* rd) { on_report(1, rd); }
static void cb_be(sm_ag_if_rd_t const* rd) { on_report(2, rd); }
static sm_cb const callbacks[N_SLICES] = {cb_embb, cb_urllc, cb_be};

/* Matching condition "S-NSSAI == (sst, sd)", as a 4-byte octet string: SST, then SD big-endian.
 * Same layout the OAI gNB decodes in capture_sst_sd (ran_func_kpm.c). */
static test_info_lst_t snssai_condition(slice_t const* s)
{
  test_info_lst_t dst = {0};
  dst.test_cond_type = S_NSSAI_TEST_COND_TYPE;
  dst.S_NSSAI = TRUE_TEST_COND_TYPE;
  dst.test_cond = ecalloc(1, sizeof(test_cond_e));
  *dst.test_cond = EQUAL_TEST_COND;
  dst.test_cond_value = ecalloc(1, sizeof(test_cond_value_t));
  dst.test_cond_value->type = OCTET_STRING_TEST_COND_VALUE;
  dst.test_cond_value->octet_string_value = ecalloc(1, sizeof(byte_array_t));
  byte_array_t* ba = dst.test_cond_value->octet_string_value;
  ba->len = 4;
  ba->buf = ecalloc(4, sizeof(uint8_t));
  ba->buf[0] = s->sst;
  ba->buf[1] = (s->sd >> 16) & 0xFF;
  ba->buf[2] = (s->sd >> 8) & 0xFF;
  ba->buf[3] = s->sd & 0xFF;
  return dst;
}

/* Action definition format 1: our three measurements, no labels, one granularity period. */
static kpm_act_def_format_1_t measurements(void)
{
  kpm_act_def_format_1_t ad = {0};
  ad.meas_info_lst_len = N_MEAS;
  ad.meas_info_lst = ecalloc(N_MEAS, sizeof(meas_info_format_1_lst_t));
  for (size_t i = 0; i < N_MEAS; i++) {
    meas_info_format_1_lst_t* m = &ad.meas_info_lst[i];
    m->meas_type.type = NAME_MEAS_TYPE;
    m->meas_type.name = cp_str_to_ba(meas_names[i]);
    m->label_info_lst_len = 1;
    m->label_info_lst = ecalloc(1, sizeof(label_info_lst_t));
    m->label_info_lst[0].noLabel = ecalloc(1, sizeof(enum_value_e));
    *m->label_info_lst[0].noLabel = TRUE_ENUM_VALUE;
  }
  ad.gran_period_ms = period_ms;
  ad.cell_global_id = NULL;
  ad.meas_bin_range_info_lst_len = 0;
  ad.meas_bin_info_lst = NULL;
  return ad;
}

/* Subscription = periodic event trigger + one style-4 action (matching condition + measurements). */
static kpm_sub_data_t slice_subscription(slice_t const* s)
{
  kpm_sub_data_t sub = {0};
  sub.ev_trg_def.type = FORMAT_1_RIC_EVENT_TRIGGER;
  sub.ev_trg_def.kpm_ric_event_trigger_format_1.report_period_ms = period_ms;
  sub.sz_ad = 1;
  sub.ad = ecalloc(1, sizeof(kpm_act_def_t));
  sub.ad[0].type = FORMAT_4_ACTION_DEFINITION;
  sub.ad[0].frm_4.matching_cond_lst_len = 1;
  sub.ad[0].frm_4.matching_cond_lst = ecalloc(1, sizeof(matching_condition_format_4_lst_t));
  sub.ad[0].frm_4.matching_cond_lst[0].test_info_lst = snssai_condition(s);
  sub.ad[0].frm_4.action_def_format_1 = measurements();
  return sub;
}

/* Does the node advertise KPM report style 4 (UE-level, condition-based)? */
static bool has_style_4(kpm_ran_function_def_t const* def)
{
  for (size_t j = 0; j < def->sz_ric_report_style_list; j++)
    if (def->ric_report_style_list[j].report_style_type == STYLE_4_RIC_SERVICE_REPORT)
      return true;
  return false;
}

int main(int argc, char* argv[])
{
  setvbuf(stdout, NULL, _IOLBF, 0);  // one JSON line at a time through `docker logs`
  fr_args_t args = init_fr_args(argc, argv);
  init_xapp_api(&args);
  sleep(1);

  e2_node_arr_xapp_t nodes = e2_nodes_xapp_api();
  defer({ free_e2_node_arr_xapp(&nodes); });
  if (nodes.len == 0) {
    printf("{\"event\":\"no_e2_node\"}\n");
    return 1;
  }

  int const KPM_ran_function = 2;
  // The testbed has one gNB; with several, each would get its own three subscriptions.
  sm_ans_xapp_t handles[N_SLICES] = {0};
  e2_node_connected_xapp_t* n = &nodes.n[0];
  kpm_ran_function_def_t const* def = NULL;
  for (size_t k = 0; k < n->len_rf; k++)
    if (n->rf[k].id == KPM_ran_function && n->rf[k].defn.type == KPM_RAN_FUNC_DEF_E)
      def = &n->rf[k].defn.kpm;
  if (def == NULL || !has_style_4(def)) {
    printf("{\"event\":\"no_kpm_style_4\"}\n");
    return 1;
  }

  for (size_t s = 0; s < N_SLICES; s++) {
    kpm_sub_data_t sub = slice_subscription(&slices[s]);
    handles[s] = report_sm_xapp_api(&n->id, KPM_ran_function, &sub, callbacks[s]);
    printf("{\"event\":\"subscribed\",\"slice\":\"%s\",\"sst\":%u,\"sd\":\"%06x\",\"ok\":%s}\n",
           slices[s].name, slices[s].sst, slices[s].sd, handles[s].success ? "true" : "false");
    free_kpm_sub_data(&sub);
  }

  xapp_wait_end_api();  // returns on SIGINT / SIGTERM (docker stop)

  for (size_t s = 0; s < N_SLICES; s++)
    if (handles[s].success)
      rm_report_sm_xapp_api(handles[s].u.handle);
  while (try_stop_xapp_api() == false)
    usleep(1000);
  printf("{\"event\":\"stopped\"}\n");
  return 0;
}
