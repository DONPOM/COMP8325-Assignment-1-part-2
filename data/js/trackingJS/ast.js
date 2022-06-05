
function dyGetTopIframe(win) {
    try {
        if (win.parent != win && win.parent.location.href) {
            return dyGetTopIframe(win.parent);
        }
    } catch (e) {}
    return win;
}
var dyTop = dyGetTopIframe(window);
(function(window, document) {
    try {
        if (!dyTop.DY) 
            dyTop.DY = {};
        DY = dyTop.DY;
        if (!dyTop.DYWork) 
            dyTop.DYWork = {};
        DYWork = dyTop.DYWork;
        if (!dyTop.$dy) 
            dyTop.$dy = function() {};
        $dy = dyTop.$dy;
        DY.slim = '';
        DY.dyid = '7421932221683727514';
        DY.server = 'dynamicyield.com';
        DY.color = 'tg.fst.';
        DY.vu = false;
        DY.aud = '';
        DY.audCHC = {};
        DY.audTCHC = {};
        DY.audTAuds = [];
        DY.audYCHC = {};
        DY.audYAuds = [];
        DY.audSCHC = {};
        DY.audLDART = '2014-11-30 09:51:33';
        DY.geoCode = 'AU';
        DY.geoCont = 'OC';
        DY.session = '535078cac4207cfddb0c4496fd7066c3';
        DY.section = '8765260';
        DY.secAct = false;
        DY.ignore = false;
        DY.layout = 'def';
        DY.nu = true;
        DY.adSelectors = [{
            slot: 20397,
            selector: "#allClassifieds",
            type: 'dyother'
        }, {
            slot: 20395,
            selector: "#autos",
            type: 'dyother'
        }, {
            slot: 20398,
            selector: "#HPMiddle3",
            type: 'dyperf'
        }, {
            slot: 20401,
            selector: "#Right1",
            type: 'dyperf'
        }, {
            slot: 20402,
            selector: "#Box2",
            type: 'dyperf'
        }, {
            slot: 17152,
            selector: "#collection-tray-item-0",
            type: 'dyother'
        }, {
            slot: 17153,
            selector: "#collection-tray-item-1",
            type: 'dyother'
        }, {
            slot: 17162,
            selector: "#collection-tray-item-10",
            type: 'dyother'
        }, {
            slot: 17163,
            selector: "#collection-tray-item-11",
            type: 'dyother'
        }, {
            slot: 17164,
            selector: "#collection-tray-item-12",
            type: 'dyother'
        }, {
            slot: 17154,
            selector: "#collection-tray-item-2",
            type: 'dyother'
        }, {
            slot: 17155,
            selector: "#collection-tray-item-3",
            type: 'dyother'
        }, {
            slot: 17156,
            selector: "#collection-tray-item-4",
            type: 'dyother'
        }, {
            slot: 17157,
            selector: "#collection-tray-item-5",
            type: 'dyother'
        }, {
            slot: 17158,
            selector: "#collection-tray-item-6",
            type: 'dyother'
        }, {
            slot: 17159,
            selector: "#collection-tray-item-7",
            type: 'dyother'
        }, {
            slot: 17160,
            selector: "#collection-tray-item-8",
            type: 'dyother'
        }, {
            slot: 17161,
            selector: "#collection-tray-item-9",
            type: 'dyother'
        }, {
            slot: 20399,
            selector: ".marketing-module-legacy",
            type: 'dyother'
        }, {
            slot: 17146,
            selector: "#expanding-nav",
            type: 'dyother'
        }, {
            slot: 17138,
            selector: "#growl_box1",
            type: 'dyother'
        }, {
            slot: 20396,
            selector: "#jobMarket",
            type: 'dyother'
        }, {
            slot: 20389,
            selector: "#wsodMarkets",
            type: 'dyother'
        }, {
            slot: 17148,
            selector: "#masthead-info",
            type: 'dyother'
        }, {
            slot: 17147,
            selector: "#masthead-tools",
            type: 'dyother'
        }, {
            slot: 20388,
            selector: "#HPMiddle",
            type: 'dyperf'
        }, {
            slot: 17144,
            selector: "#adx-middleright",
            type: 'dyperf'
        }, {
            slot: 20403,
            selector: "#mostPopWidget",
            type: 'dyother'
        }, {
            slot: 17141,
            selector: "#image_back #clickbox",
            type: 'dyother'
        }, {
            slot: 17139,
            selector: "#image_back #gw_HDLink",
            type: 'dyother'
        }, {
            slot: 17142,
            selector: "#image_back #box3 img",
            type: 'dyother'
        }, {
            slot: 17149,
            selector: "#image_back #textbox a",
            type: 'dyother'
        }, {
            slot: 20394,
            selector: "#realEstate",
            type: 'dyother'
        }, {
            slot: 20400,
            selector: ".timeswireModule",
            type: 'dyother'
        }, {
            slot: 17145,
            selector: "#adx-1",
            type: 'dyperf'
        }, {
            slot: 20387,
            selector: "#Top",
            type: 'dyperf'
        }, {
            slot: 20392,
            selector: "#TopLeft",
            type: 'dyperf'
        }, {
            slot: 20393,
            selector: "#TopRight",
            type: 'dyperf'
        }, {
            slot: 20386,
            selector: "#extendedVideoPlayerModule",
            type: 'dyother'
        }, ];
        DY.adUrlSelectors = [{
            slot: 20183, selector: ".columnGroup.fullWidth.flushBottom:eq(1)", url: "www.nytimes.com\/", urlMatchType: "ex", type: 'dycontent', aSelector: '.story'
        }, {
            slot: 20182, selector: "#insideNYTimesBrowser", url: "www.nytimes.com\/", urlMatchType: "ex", type: 'dycontent', aSelector: '.story'
        }, {
            slot: 20180, selector: ".a-column", url: "www.nytimes.com\/", urlMatchType: "ex", type: 'dycontent', aSelector: 'article'
        }, {
            slot: 20181, selector: ".b-column", url: "www.nytimes.com\/", urlMatchType: "ex", type: 'dycontent', aSelector: 'article'
        }, {
            slot: 20390, selector: "#cColumnTopSpanRegion", url: "www.nytimes.com\/", urlMatchType: "ex", type: 'dycontent', aSelector: '.story'
        }, {
            slot: 20184, selector: "#wellRegion", url: "www.nytimes.com\/", urlMatchType: "ex", type: 'dycontent', aSelector: '.headlinesOnly'
        }, {
            slot: 20185, selector: "#timesMinuteContainer", url: "www.nytimes.com\/", urlMatchType: "ex", type: 'dycontent', aSelector: 'li'
        }, {
            slot: 20186, selector: ".bColumn.column:eq(0)", url: "www.nytimes.com\/", urlMatchType: "ex", type: 'dycontent', aSelector: '.story'
        }, ];
        DY.svars = [{
            id: 63,
            script: "function dyGetMetaTag(tagProp,tagName){try{var metaTag=document.querySelector(\"meta[\"+tagProp+\"='\"+tagName+\"']\");if(metaTag!=null&&typeof metaTag.content!=='undefined'&&metaTag.content!==''){return metaTag.content.toLowerCase()}else{return ''}}catch(ignore){}}\r\nDY.segdef = dyGetMetaTag('name','CG');"
        }, {
            id: 64,
            script: "function dyGetMetaTag(tagProp,tagName){try{var metaTag=document.querySelector(\"meta[\"+tagProp+\"='\"+tagName+\"']\");if(metaTag!=null&&typeof metaTag.content!=='undefined'&&metaTag.content!==''){return metaTag.content.toLowerCase()}else{return ''}}catch(ignore){}}\r\ndyGetMetaTag('name','SCG');"
        }, {
            id: 65,
            script: "function dyGetMetaTag(tagProp,tagName){try{var metaTag=document.querySelector(\"meta[\"+tagProp+\"='\"+tagName+\"']\");if(metaTag!=null&&typeof metaTag.content!=='undefined'&&metaTag.content!==''){return metaTag.content.toLowerCase()}else{return ''}}catch(ignore){}}\r\ndyGetMetaTag('name','PST');"
        }, ];
        DY.goalScripts = [];
        DY.audienceRules = [{
            "audience": 910,
            "updatedAt": "2014-04-13 08:36:46",
            "rule": [{
                "condType": "PageViews",
                "conds": [{
                    "id": 1410,
                    "parameter": null,
                    "selectMethod": "count",
                    "selectParameter": null,
                    "hitCountMethod": ">=",
                    "hitCount": 20
                }
                ]
            }
            ],
            "session": 0,
            "hidden": 0
        }, {
            "audience": 1757,
            "updatedAt": "2014-08-27 20:01:55",
            "rule": [{
                "condType": "SiteVariable",
                "conds": [{
                    "id": 2761,
                    "parameter": 63,
                    "selectMethod": "equals",
                    "selectParameter": "opinion",
                    "hitCountMethod": ">=",
                    "hitCount": 3
                }
                ]
            }
            ],
            "session": 0,
            "hidden": 0
        }, {
            "audience": 1758,
            "updatedAt": "2014-08-27 20:14:27",
            "rule": [{
                "condType": "UnitClick",
                "conds": [{
                    "id": 2762,
                    "parameter": 20386,
                    "selectMethod": "equals",
                    "selectParameter": null,
                    "hitCountMethod": ">=",
                    "hitCount": 3
                }
                ]
            }, {
                "condType": "SiteVariable",
                "conds": [{
                    "id": 2763,
                    "parameter": 63,
                    "selectMethod": "equals",
                    "selectParameter": "video",
                    "hitCountMethod": ">=",
                    "hitCount": 3
                }
                ]
            }
            ],
            "session": 0,
            "hidden": 0
        }, {
            "audience": 2765,
            "updatedAt": "2014-11-28 16:28:40",
            "rule": [{
                "condType": "PageVisited",
                "conds": [{
                    "id": 4667,
                    "parameter": null,
                    "selectMethod": "contains",
                    "selectParameter": "\/movies\/",
                    "hitCountMethod": ">=",
                    "hitCount": 1
                }
                ]
            }
            ],
            "session": 0,
            "hidden": 0
        }
        ];
        DY.fpExp = [];
        DY.prd = [];
        DY.tzo = - 5;
        var dys = document.createElement("script");
        dys.setAttribute("src", "http://cdn.dynamicyield.com/scripts/6217/dy-min.js?v=3635");
        dys.setAttribute("type", "text/javascript");
        dys.setAttribute("async", "true");
        var dyjqs = window.document.createElement('script');
        dyjqs.type = 'text/javascript';
        dyjqs.src = 'http://cdn.dynamicyield.com/scripts/6217/dyjq-min.js?v=3635';
        dyjqs.async = true;
        var head = document.getElementsByTagName('head')[0];
        head.appendChild(dyjqs);
        document.body.appendChild(dys);
    } catch (exception) {}
})(window.parent, window.parent.document);
try {
    if (typeof window.parent.DYO !== "undefined") {
        var oldSetVar = typeof window.parent.DYO.setVariations !== "undefined";
        var newSetVar = typeof window.parent.DYO.storeAttributeVariations !== "undefined";
        if (oldSetVar || newSetVar) {
            var setVarFunc = newSetVar ? window.parent.DYO.storeAttributeVariations : window.parent.DYO.setVariations;
        }
    }
} catch (e) {}
