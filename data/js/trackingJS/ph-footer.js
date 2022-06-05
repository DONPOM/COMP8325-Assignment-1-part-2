var ph_bar = new PornhubNetworkBar("PornhubNetworkBar", "pornhub", networkSegement, false, "body { background-position: 0 24px;} div#starbust-header { background-position: center -4px; }div.bar_body div.bar_items { width:960px;} " + networkQuery, false);
if (!platformPcSet) {
    $j(function() {
        if (window.canRunAds === undefined && shouldRun()) {
            var b = document.createElement("a");
            b.href = $j(".pre-footer iframe").attr("src");
            var a = getUri(adServer + "?" + b.hash.substring(1));
            $j('<div style="padding: 30px 0 0;margin: 0 auto;width: 950px;text-align: center;"><iframe style="display:block;margin:0 auto;" width="950" height="250" frameborder="0" scrolling="no" src="http://' + a + '"></iframe></div>').insertAfter(".pre-footer")
        }
    })
};
