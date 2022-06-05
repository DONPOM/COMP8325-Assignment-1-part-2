
                (function() {
                    var s = document.createElement('script');
                    var el = document.getElementsByTagName('script')[0];
                                            s.src = ('https:' == document.location.protocol ? 'https://s' : 'http://l') + '.yimg.com/ss/rapid-3.18.1.js';
                                        s.onload = function(){
                        var YAHOO = window.YAHOO;
                        if (YAHOO) {
                            var keys = {
                                pd:'/dashboard',
                                _li:1,
                                b_id:0,
                                i_rad:0,
                                i_strm:0
                            };
                            var conf = {
                                                                spaceid:1197716038,
                                client_only:1,
                                yql_enabled:false,
                                keys:keys,
                                nol:1
                            };
                            YAHOO.rapid = new YAHOO.i13n.Rapid(conf);
                        }
                    };
                    el.parentNode.insertBefore(s, el);
                })();
            