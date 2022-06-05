//tealium universal tag - utag.431 ut4.0.201411261918, Copyright 2014 Tealium.com Inc. All Rights Reserved.
try {
    (function(id, loader, u) {
        try {
            u = utag.o[loader].sender[id] = {}
        } catch (e) {
            u = utag.sender[id]
        };
        u.ev = {
            'view': 1
        };
        u.qsp_delim = "&";
        u.kvp_delim = "=";
        u.qs_delim = "?";
        u.tag_type = "img";
        u.base_url = "http://leadback.advertising.com/adcedge/lb";
        u.static_params = "site=695501&srvc=1";
        u.map = {
            "ret_aol_product": "betr"
        };
        u.extend = [function(a, b, c, d, e, f, g) {
            d = b['dom.pathname'];
            if (typeof d == 'undefined')
                return;
            c = [{
                'office-365.aspx': '41971=958879[720]'
            }, {
                'domain-name-search.aspx': '41967=958871[720]'
            }, {
                'website-builder.aspx': '41968=958873[720]'
            }, {
                'web-hosting.aspx': '41969=958875[720]'
            }, {
                'wordpress-hosting.aspx': '41970=958877[720]'
            }, {
                'online-business.aspx': '41977=958891[720]'
            }, {
                'ssl.aspx': '41973=958883[720]'
            }, {
                'servers': '41972=958881[720]'
            }, {
                'searchresults.aspx': '41967=958871[720]'
            }, {
                'online-store.aspx': '41976=958889[720]'
            }, {
                'business-marketing.aspx': '41974=958885[720]'
            }, {
                'web-design.aspx': '41975=958887[720]'
            }
            ];
            var m = false;
            for (e = 0; e < c.length; e++) {
                for (f in c[e]) {
                    if (d.toString().indexOf(f)>-1) {
                        b['ret_aol_product'] = c[e][f];
                        m = true
                    };
                };
                if (m)
                    break
            };
            if (!m)
                b['ret_aol_product'] = '41227=945793[720] ';
        }
        ];
        u.send = function(a, b) {
            if (u.ev[a] || typeof u.ev.all != "undefined") {
                for (c = 0; c < u.extend.length; c++) {
                    try {
                        d = u.extend[c](a, b);
                        if (d == false)
                            return 
                    } catch (e) {}
                };
                var c, d, e, f;
                c = [];
                for (d in utag.loader.GV(u.map)) {
                    if (typeof b[d] != "undefined" && b[d] != "") {
                        e = u.map[d].split(",");
                        for (f = 0; f < e.length; f++) {
                            if (e[f] == "qsp_delim" || e[f] == "kvp_delim" || e[f] == "qs_delim" || e[f] == "base_url" || e[f] == "secure_base_url") {
                                u[e[f]] = b[d];
                            } else {
                                c.push(e[f] + u.kvp_delim + encodeURIComponent(b[d]));
                            }
                        }
                    }
                }
                u.secure_base_url = u.secure_base_url || "https://secure.leadback.advertising.com/adcedge/lb" || u.base_url;
                u.url = (location.protocol == "https:" ? u.secure_base_url : u.base_url);
                if (u.url.indexOf("http") != 0 && u.url.indexOf("/") != 0) {
                    u.url = "//" + u.url;
                }
                if (u.url.indexOf(u.qs_delim) < 0 && (c.length > 0 || u.static_params.length > 0)) {
                    u.url += u.qs_delim
                }
                if (u.static_params) {
                    if (c.length > 0) {
                        u.url += u.static_params + u.qsp_delim;
                    } else {
                        u.url += u.static_params;
                    }
                }
                if (u.tag_type == "img") {
                    u.img = new Image();
                    u.img.src = u.url + c.join(u.qsp_delim);
                } else if (u.tag_type == "script") {
                    u.s = document.getElementsByTagName("script")[0];
                    u.scr = document.createElement("script");
                    u.scr.type = "text/javascript";
                    u.scr.src = u.url + c.join(u.qsp_delim);
                    u.s.parentNode.insertBefore(u.scr, u.s);
                } else {
                    d = document.createElement("iframe");
                    d.setAttribute('id', '431');
                    d.setAttribute('height', '1');
                    d.setAttribute('width', '1');
                    d.setAttribute('style', 'display:none');
                    d.setAttribute('src', u.url + c.join(u.qsp_delim));
                    document.body.appendChild(d);
                }
            }
        }
        try {
            utag.o[loader].loader.LOAD(id)
        } catch (e) {
            utag.loader.LOAD(id)
        }
    })('431', 'godaddy.godaddy');
} catch (e) {}

