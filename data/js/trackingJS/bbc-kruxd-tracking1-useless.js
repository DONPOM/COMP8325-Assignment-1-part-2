
                /*<![CDATA[*/
                if (bbcdotcom.data.a == 1 || bbcdotcom.data.b == 1) {
                    window.Krux||((Krux=function(){Krux.q.push(arguments);}).q=[]);
                    (function(){
                        function  retrieve(n){
                            var  m,  k='kx'+n;
                            if  (window.localStorage)  {
                                return  window.localStorage[k]  ||  "";
                            }  else  if  (navigator.cookieEnabled)  {
                                m  =  document.cookie.match(k+'=([^;]*)');
                                return  (m  &&  unescape(m[1]))  ||  "";
                            }  else  {
                                return  '';
                            }
                        }
                        Krux.user  =  retrieve('user');
                        Krux.segments  =  retrieve('segs')  ?  retrieve('segs').split(',')  :  [];
                        //  DFP  Premium
                        var  dfpp  =  [];
                        if  (Krux.user)  {
                            dfpp.push('khost='  +  encodeURIComponent(location.hostname));
                            dfpp.push('kuid='  +  Krux.user);
                        }
                        for  (var  i  =  0;  i  <  Krux.segments.length;  i++  )  {
                            dfpp.push('ksg='  +  Krux.segments[i]);
                        }
                        Krux.dfppKeyValues  =  dfpp.length  ?  dfpp.join(';')  +  ';'  :  '';
                    })();
                }
                /*]]>*/
            