
//<![CDATA[
(function(w, d) {
        if (w.rb_counter) return;
        function h() {
                var n, a = arguments;
                for(var i=a.length;i--;){
                        n = a[i].split(',');
                        for(var j=n.length;j--;){
                                c(n[j]);
                        }
                }
        }
        function c(n) {
                var m, r = parseInt(Math.random()*1E9), s;
                if (!(m = n.match(/^(?:cl([bn])|([adgin]))(\d+)(?:sz(\d+))?/))) return;
                if (m[1]) {
                        s = "s" + m[1];
                }
                else if(m[2] == "n") {
                        s = "nc";
                }
                else if(m[2]=="i") {
                        r = null;
                        s = m[2];
                }
                else {
                        s = m[2];
                }
                s += m[3] + ".gif?";
                if (m[4]) s += "sz=" + m[4];
                if (r) s += "&rnd=" + r;
                (new Image).src = "//rs.mail.ru/" + s;
        }
        (function(o, e, fn) {
                if (o.addEventListener) o.addEventListener(e, fn, false);
                else if (o.attachEvent) o.attachEvent('on' + e, fn);
                else o['on' + e] = fn;
        })(d, 'mousedown', function(e) {
                var n;
                e = e || w.event;
                e = e.target || e.srcElement;
                while (e.parentNode){
                        if ((n = e.getAttribute('name')) && (n = n.toString())){
                                h(n);
                        }
                        e = e.parentNode;
                }
        });
        w.rb_counter = h;
})(window, document);
//]]>
