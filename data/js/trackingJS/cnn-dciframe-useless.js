
var jsmd=_jsmd.init(),pageURL=location.href.toLowerCase();
if (pageURL.indexOf("/.element/ssi/ads.iframes/")==-1&&pageURL.indexOf("/doubleclick/dartiframe.html")==-1&&pageURL.indexOf("/search/")==-1){
if (_jsmd.plugin.gQuery("refresh")){
jsmd.trackMetrics("dynamic-autoRefresh","autorefresh","cnn-autorefresh");
} else if (_jsmd.plugin.gQuery("is_LR")){
} else if (cnn_metadata.template_type_content!="gallery"){
jsmd.send();
}
}