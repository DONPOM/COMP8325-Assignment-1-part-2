var _gaq = _gaq || [];
            _gaq.push(['_setAccount', 'UA-97144-8']);
            _gaq.push(['_setAllowLinker', true]);
            _gaq.push(['_setDomainName', 'tumblr.com']);
                            _gaq.push(['_setCustomVar', 2, 'User_Logged_In', 'Yes', 3]);
                _gaq.push(['_trackPageview', '/dashboard?route=%2Fdashboard']);
            
            (function() {
                var s = document.createElement('script');
                s.async = true;
                var el = document.getElementsByTagName('script')[0];
                s.src = ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + '.google-analytics.com/ga.js';
                el.parentNode.insertBefore(s, el);
            })();

            // Google Universal Analytics
            (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
                (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
                m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
            })(window,document,'script','//www.google-analytics.com/analytics.js','ga');
            ga('create', 'UA-97144-14', 'tumblr.com', {
                'allowLinker': true,
                'sampleRate': 0.7
            });
                            ga('set', 'dimension2', 'Yes');
                ga('send', 'pageview', '/dashboard?route=%2Fdashboard');