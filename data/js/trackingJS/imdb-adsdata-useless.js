
doWithAds(function(){
generic.monitoring.record_metric("ads_js_request_to_done", (new Date().getTime()) - window.ads_js_start);
generic.monitoring.set_forester_info("homepage");
generic.monitoring.set_twilight_info(
"homepage",
"AU",
"541e6a8fb93c8e9c767e5cbdec79546f4f36d6b6",
"2014-12-03T04%3A02%3A37GMT",
"http://s.media-imdb.com/twilight/?",
"consumer");
generic.send_csm_head_metrics && generic.send_csm_head_metrics();
generic.monitoring.start_timing("page_load");
generic.seconds_to_midnight = 14243;
generic.days_to_midnight = 0.1648495346307754;
ad_utils.set_slots_on_page({ 'injected_navstrip':1, 'top_rhs':1, 'top_ad':1, 'injected_billboard':1, 'rhs_cornerstone':1 });
custom.full_page.data_url = "http://ia.media-imdb.com/images/G/01/imdbads/js/graffiti_data-3955167245._CB318919114_.js";
consoleLog('advertising initialized','ads');
},"ads js missing, skipping ads setup.");
var _gaq = _gaq || [];
_gaq.push(['_setCustomVar', 4, 'ads_abtest_treatment', 'b']);
