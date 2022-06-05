(function() {
    var d = {
        create: function(t, u, v) {
            if (v) {
                var s = new Date();
                s.setTime(s.getTime() + (v * 24 * 60 * 60 * 1000));
                var a = "; expires=" + s.toGMTString()
            } else {
                var a = ""
            }
            document.cookie = t + "=" + u + a + "; path=/"
        },
        read: function(s) {
            var u = s + "=";
            var a = document.cookie.split(";");
            for (var t = 0; t < a.length; t++) {
                var v = a[t];
                while (v.charAt(0) == " ") {
                    v = v.substring(1, v.length)
                }
                if (v.indexOf(u) == 0) {
                    return v.substring(u.length, v.length)
                }
            }
            return null
        },
        erase: function(a) {
            this.create(a, "", - 1)
        }
    };
    var q = new Date(2014, 9 - 1, 16);
    var c = new Date();
    if (c > q) {
        return 
    }
    if (d.read("prv-shwn") == "1") {
        return 
    } else {
        d.create("prv-shwn", "1", 300)
    }
    var n = function() {
        document.body.removeChild(o);
        document.body.className = document.body.className.replace("privacy-policy", "")
    };
    var i = function() {
        if (document.domain.indexOf("travel.aol.com")>-1) {
            return "aol"
        } else {
            return "mq"
        }
    };
    var k = {
        mq: {
            text: "MapQuest/AOL's Privacy Policy will be updated on 9/15/14. By continuing to use MapQuest, you agree to these updates.",
            url: "http://privacy.aol.com/faq?icid=mq_bannerpolicy"
        },
        aol: {
            text: "Our Terms and Privacy Policy will be updated 9/15/14. By continuing to use AOL, you agree to the updates and to arbitration of disputes.",
            url: "http://privacy.aol.com/faq"
        }
    };
    var b = i();
    var o = document.createElement("table");
    var h = document.createElement("tbody");
    o.className = "privacy-table";
    o.style.cssText = "color:#FFF;width:100%;position:relative;left:0;top:0;z-index:100000000;height:32px;background:#2991F9;";
    var j = document.createElement("tr");
    var g = document.createElement("td");
    g.innerHTML = k[b].text;
    g.style.cssText = "font-size:12px;font-family:sans-serif;text-align:center;vertical-align:middle;";
    var f = document.createElement("span");
    f.style.cssText = "white-space:nowrap;";
    var m = document.createElement("a");
    m.innerHTML = "More.";
    m.target = "_blank";
    m.href = k[b].url;
    m.style.cssText = "color:#FFF;text-decoration:underline;margin-left:8px;";
    var p = document.createElement("table");
    var e = document.createElement("tbody");
    var r = document.createElement("tr");
    p.onclick = n;
    p.style.cssText = "background:#2169CF;margin:0 0 0 8px;padding:0;border:none;cursor:pointer;display: inline-block;*display: inline;zoom: 1;width:20px;border-collapse: collapse;height:20px;vertical-align:middle;position:relative;";
    var l = document.createElement("td");
    l.innerHTML = "&times;";
    l.style.cssText = "color:#FFF;font-size:20px;font-family:Arial,sans-serif;line-height:20px;text-align:center;padding:0;height:20px;width:20px;";
    r.appendChild(l);
    e.appendChild(r);
    p.appendChild(e);
    f.appendChild(m);
    f.appendChild(p);
    g.appendChild(f);
    j.appendChild(g);
    h.appendChild(j);
    o.appendChild(h);
    document.body.insertBefore(o, document.body.childNodes[0]);
    document.body.className += " privacy-policy"
})();
