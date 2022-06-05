
        var googletag = googletag || {},
            gptadslots = [];
        googletag.cmd = googletag.cmd || [];

        (function() {
            var gads = document.createElement('script'),
                useSSL = 'https:' == document.location.protocol,
                node;
            gads.async = true;
            gads.type = 'text/javascript';
            gads.src = (useSSL ? 'https:' : 'http:') + '//www.googletagservices.com/tag/js/gpt.js';
            node = document.getElementsByTagName('script')[0];
            node.parentNode.insertBefore(gads, node);
        })();

        googletag.cmd.push(function() {
                                                                    gptadslots[0] = googletag.defineSlot('/54549544/homepage_logged_in', [[300,250]], 'div-gpt-ad-783291444').setTargeting('p', ['ATF']);

                    
                    gptadslots[0].addService(googletag.pubads());
                            
            googletag.pubads().setTargeting('sec', ['true']);
            googletag.pubads().enableAsyncRendering();
            googletag.enableServices();
        });
    