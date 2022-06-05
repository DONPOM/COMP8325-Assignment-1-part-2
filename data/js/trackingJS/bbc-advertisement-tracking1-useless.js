                 /*<![CDATA[*/
            if (bbcdotcom.data.a == 1 || bbcdotcom.data.b == 1) {
                                BBC.adverts.setData(bbcdotcom.data);
                bbcdotcom.config.setAdsEnabled(bbcdotcom.data.a);
                bbcdotcom.config.setAnalyticsEnabled(bbcdotcom.data.b);
            }
            if (bbcdotcom.data.a == 1) {
                                (function(){
                    var zoneFile = "3pt_zone_file";

                    bbcdotcom.objects('page', 'create', bbcdotcom);
                    
                                        BBC.adverts.setScriptRoot("http://static.bbci.co.uk/bbcdotcom/0.3.291/script/");
                    BBC.adverts.setImgRoot("http://static.bbci.co.uk/bbcdotcom/0.3.291/img/");
                    BBC.adverts.init({
                        domain: "www.bbc.com",
                        location: window.location.pathname,
                        zoneVersion: zoneFile,
                        zoneAuto: false,
                        siteAuto: false,
                        keywordsAuto: false,
                        zoneReferrer: document.referrer
                    });
                })();
                if(typeof bbcdotcom !== 'undefined' && typeof bbcdotcom.survey !== 'undefined' && typeof bbcdotcom.survey.init === 'function') {
                    bbcdotcom.survey.init();
                }
            }
            /*]]>*/
        