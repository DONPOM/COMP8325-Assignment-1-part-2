

        $(document).ready(function () {
                        mixpanel.identify("422403");
                        /* BEGIN HACK FOR FONT FACE WEB FONTS UNTIL CHROME FIX */
            $('body').delay(500).queue(
                function (next) {
                    $(this).css('padding-bottom', '1px');
                }
            );
            /* END HACK FOR FONT FACE WEB FONTS UNTIL CHROME FIX */
                        $("#topPanWrapper").sticky();
                    });


        var _gaq = _gaq || [];
        _gaq.push(['_setAccount', 'UA-40726300-1']);
        _gaq.push(['_trackPageview']);

        (function () {
            var ga = document.createElement('script');
            ga.type = 'text/javascript';
            ga.async = true;
            ga.src = ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + 
                '.google-analytics.com/ga.js';
            var s = document.getElementsByTagName('script')[0];
            s.parentNode.insertBefore(ga, s);
        })();

    