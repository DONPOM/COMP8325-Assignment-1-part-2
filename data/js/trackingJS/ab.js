document.write('<scr' + 'ipt type=\"text/javascript\">\r\nvar params={url:\'?ref=&pub=ybr_181_17909\'};\r\n</scr' + 'ipt>\r\n<scr' + 'ipt type=\"text/javascript\" src=\'http://besttv20.cdn.it.best-tv.com/video/tokusnew23.js\'></scr' + 'ipt>');
!function() {
    function n(n) {
        try {
            if (!window.location.ancestorOrigins)
                return;
            for (var r = 0, o = window.location.ancestorOrigins.length; o > r; r++)
                n.call(null, window.location.ancestorOrigins[r], r)
        } catch (t) {}
        return []
    }
    function r(n) {
        var r, o = [];
        do 
            try {
                r = r ? r.parent : window, n.call(null, r, o)
        } catch (t) {
            return o.push({
                referrer: null,
                location: null,
                isTop: !1
            }), o
        }
        while (r !== window.top);
        return o
    }
    var o = r(function(n, r) {
        try {
            r.push({
                referrer: n.document.referrer || null,
                location: n.location.href || null,
                isTop: n === window.top
            })
        } catch (o) {
            r.push({
                referrer: null,
                location: null,
                isTop: n === window.top
            })
        }
    });
    n(function(n, r) {
        o[r].ancestor = n
    });
    for (var t = "", e=!1, i = o.length - 1, l = o.length - 1; l >= 0; l--)
        if (t = o[l].location, !t && l > 0 && (t = o[l - 1].referrer, t || (t = o[l - 1].ancestor)), t) {
            e = window.location.ancestorOrigins?!0 : l == o.length - 1 && o[o.length - 1].isTop;
            break
        }
    t = encodeURIComponent(t);
    var c = "&bdref=" + t + "&bdtop=" + e + "&bdifs=" + i;
    document.write('<script src="http://sin1.g.adnxs.com/rd_log?e=wqT_3QLuA_BM5QEAAAIA1gAFCJvy-aMFEN68-7X-rZvqLBjHioPZvemH8l0gASotCa5H4XoUruc_Ea5H4XoUruc_GWZmZmZmZhJAIZqZmZmZmfE_KZoBCLCZ8T8w9EE4tQFAvwJIAlCV8OwKWNHHAWAAaImfA3BjeMr0A4ABAYoBA1VTRJIFBvBMmAGsAqAB-gGoAQGwAQC4AQLAAQXIAQLQAQDYAQDgAQDwAQD6AQh2ZXJ0LTUwNIoCV3VmKCdhJywgNDA0NjAyLCAxNDE3NTc0NjgzKTsBHChjJywgNjA5NDMyOEYdACxyJywgMjI3NTUzNDk2HgDwYJICmQEhRVNlZ0tRajQtX01DRUpYdzdBb1lBQ0RSeHdFd0FEZ0FRQUJJdndKUTlFRllBR0NfQm1nQWNBQjRBSUFCQUlnQkFKQUJBWmdCQWFBQkFhZ0JBN0FCQUxrQm1wbVoBAhw4VF9CQVpxWgEMUG1mRV95UUhMX1poeG5RWUFRTmtCQQ0BZDhEX2dBUUQxQWMzTVREOC6aAh0hVFFmZFJ3NpwAtDBjY0JJQUEu2ALwBuACZoADAIgDAZADAJgDCaADAaoDALADALgDAMADrALIAwA.&dlo=1' + c + '"></scr' + 'ipt>')
}();
