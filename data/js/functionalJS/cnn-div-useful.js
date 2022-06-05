$j("div.cnn_mc23x1cnntr[data-vr-zone*='3col']").attr("data-vr-zone","3collayout");
$j("div.cnn_mc23x1cnntr[data-vr-zone*='3col']").nextAll("div.cnn_mc23x1cnntr[data-vr-zone*='Top']").each(function(innerIndex){
$j(this).attr('data-vr-zone','c'+(innerIndex+1)+'Bottom');
});
$j("div.cnn_mc23x1cnntr[data-vr-zone*='3col']").nextAll("div.cnn_mc23x1cnntr[data-vr-zone*='Bottom']").each(function(innerIndex){
$j(this).attr('data-vr-contentbox','c'+(innerIndex+1)+'Bottom');
});