if (window.amznpubstudioDomainScriptLoaded) {
    if (typeof console && typeof console.log) {
        console.log('Publisher studio javascript already exists on the page. This is the second instance.');
        console.log('Please use only 1 instance of publisher studio javascript on the page');
    }
} else {
    window.amznpubstudioDomainScriptLoaded = true;
    window.amznpubstudioDomainId = "ryjoll-20#6a5a6305-9628-418e-90fb-4c3848720f82#US";
    (function() {
        var script = document.createElement( "script" );
        script.type = "text/javascript";
        script.src = "http://ps-us.amazon-adsystem.com/services/getScript?domainId=6a5a6305-9628-418e-90fb-4c3848720f82&storeId=ryjoll-20&locale=US&domainStatus=ENABLED";
        script.charset = "UTF-8";
        script.async = "async";
        (document.body || document.head).appendChild(script);
    })();
}
window.amznpubstudioAdservingSettings = {
    "auto_tagging": false,
    "slidein_wdgt_category": "All",
    "at_tagoverwrite": false,
    "slidein_wdgt": false,
    "ads_enabled": true,
    "shopnow_pos": "bottom",
    "autobot": "on"
};

