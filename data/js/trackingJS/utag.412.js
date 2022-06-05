//tealium universal tag - utag.412 ut4.0.201411071927, Copyright 2014 Tealium.com Inc. All Rights Reserved.
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
        u.domain = "godaddy";
        u.spacedesc = "16057653_1061349_1x1_1061349_1061349";
        u.db_afcr = "123";
        u.group = "GoDaddy";
        u.base_url = "//media." + u.domain + ".com/ipixel?spacedesc=" + u.spacedesc + "&db_afcr=" + u.db_afcr + "&target=_blank&group=" + u.group + "&";
        u.map = {
            "te_event": "event"
        };
        u.extend = [function(a, b, c, d, e, f, g) {
            d = b['dom.url'];
            if (typeof d == 'undefined')
                return;
            c = [{
                'domains/searchresults.aspx': 'Comp_Domain_Search'
            }, {
                'domain-name-search.aspx': 'Domain'
            }, {
                'hosting/website-builder.aspx': 'WSB'
            }, {
                'ecommerce/shopping-cart.aspx': 'Online_Store'
            }, {
                'hosting/web-hosting.aspx': 'Hosting'
            }, {
                'hosting/wordpress-hosting.aspx': 'WordPress'
            }, {
                'godaddy.com/servers': 'Servers'
            }, {
                'godaddy.com/ssl.aspx': 'SSL'
            }, {
                'products/business-marketing.aspx': 'Get_Found'
            }, {
                'business/office-365.aspx': 'Email'
            }, {
                'offers/online-business.aspx': 'GOT'
            }, {
                'design/web-design.aspx': 'PrrofessionalServices'
            }, {
                'Order_Confirmation.aspx': 'OrderConfirmation'
            }, {
                'godaddy.com/Basket.aspx': 'Cart'
            }, {
                'godaddy.com/Checkout/Payment.aspx': 'Payments'
            }, {
                'godaddy.com/Checkout/AccountReview.aspx': 'Account_Review'
            }, {
                'ecommerce/online-store.aspx': 'Online_Store'
            }
            ];
            var m = false;
            for (e = 0; e < c.length; e++) {
                for (f in c[e]) {
                    if (d.toString().indexOf(f)>-1) {
                        b['te_event'] = c[e][f];
                        m = true
                    };
                };
                if (m)
                    break
            };
            if (!m)
                b['te_event'] = 'HomePage';
        }
        ];
        u.send = function(a, b, c, d, e, f) {
            if (u.ev[a] || typeof u.ev.all != "undefined") {
                for (c = 0; c < u.extend.length; c++) {
                    try {
                        d = u.extend[c](a, b);
                        if (d == false)
                            return 
                    } catch (e) {}
                };
                c = [];
                for (d in utag.loader.GV(u.map)) {
                    if (typeof b[d] != "undefined" && b[d] != "") {
                        e = u.map[d].split(",");
                        for (f = 0; f < e.length; f++) {
                            c.push(e[f] + u.kvp_delim + encodeURIComponent(b[d]))
                        }
                    }
                }
                if (typeof b._corder != "undefined" && b._corder) {
                    c.push('revenue=' + b._ctotal);
                    c.push('x_orderid=' + b._corder);
                } else {
                    c.push('revenue=REVENUE');
                }
                c.push('random=' + Math.random());
                d = document.createElement("iframe");
                d.setAttribute('id', '412');
                d.setAttribute('height', '1');
                d.setAttribute('width', '1');
                d.setAttribute('style', 'display:none');
                d.setAttribute('src', u.base_url + c.join(u.qsp_delim));
                document.body.appendChild(d);
            }
        }
        try {
            utag.o[loader].loader.LOAD(id)
        } catch (e) {
            utag.loader.LOAD(id)
        }
    })('412', 'godaddy.godaddy');
} catch (e) {}

