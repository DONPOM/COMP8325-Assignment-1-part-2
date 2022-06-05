
            /*<![CDATA[*/
            bbcdotcom.objects('bbcdotcom.config', 'create');
            bbcdotcom.config =(function(ads, analytics) {
                var adsEnabled=ads,
                    analyticsEnabled=analytics,
                    jsPrefix="http://static.bbci.co.uk/bbcdotcom/0.3.291/script",
                    swfPrefix="http://static.bbci.co.uk/bbcdotcom/0.3.291/swf",
                    cssPrefix="http://static.bbci.co.uk/bbcdotcom/0.3.291/style";
                return {
                    setAdsEnabled: function(enabled) {
                        /* Once it has been disable to not allow it to be enabled */
                        adsEnabled = (adsEnabled !== 0) ? enabled : 0;
                    },
                    isAdsEnabled: function() {
                        return adsEnabled;
                    },
                    setAnalyticsEnabled: function(enabled) {
                        /* Once it has been disable to not allow it to be enabled */
                        analyticsEnabled = (analyticsEnabled !== 0) ? enabled : 0;
                    },
                    isAnalyticsEnabled: function() {
                        return analyticsEnabled;
                    },
                    setJsPrefix: function (prefix) {
                        jsPrefix = prefix;
                    },
                    getJsPrefix: function () {
                        return jsPrefix;
                    },
                    setSwfPrefix: function (prefix) {
                        swfPrefix = prefix;
                    },
                    getSwfPrefix: function () {
                        return swfPrefix;
                    },
                    setCssPrefix: function (prefix) {
                        cssPrefix = prefix;
                    },
                    getCssPrefix: function () {
                        return cssPrefix;
                    }
                };
            }(1, 1));
            bbcdotcom.objects('bbcdotcom.siteCatalyst', 'create');
            bbcdotcom.siteCatalyst = {"ch":"","cdp":"3","ce":"UTF-8"};
            bbcdotcom.objects('bbcdotcom.stats', 'create');
            /* Create BBC.adverts skeleton */
            if (window.BBC === undefined) {
                var BBC = {};
            }
            if (window.BBC.adverts === undefined) {
                BBC.adverts = {
                    setZone: function () {},
                    configure: function () {},
                    write: function () {},
                    show: function () {},
                    isActive: function () {
                        return false;
                    },
                    setScriptRoot: function () {},
                    setImgRoot: function () {},
                    getAdvertTag: function () {},
                    getSectionPath: function() {},
                    showPartnerButtons: function() {}
                };
            }
            (function(){
                if(typeof require !== 'undefined') {
                    require({
                        paths:{
                            "bbcdotcom":"http://static.bbci.co.uk/bbcdotcom/0.3.291/script"
                        }
                    });
                }
            })();
            if (typeof orb !== 'undefined' && typeof orb.fig === 'function') {
                bbcdotcom.data = {
                    a: orb.fig('ad')? 1 : 0,
                    b: (0 == orb.fig('uk')) ? 1 : 0,
                    c: orb.fig('ap')
                };
            } else {
                document.write('<script type="text/javascript" src="http://tps.bbc.com/wwscripts/data">\x3C/script>');
            }
            /*]]>*/
        