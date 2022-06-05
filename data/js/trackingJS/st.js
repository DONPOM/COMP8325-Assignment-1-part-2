
try {
    DY = typeof DY === 'undefined' ? new Object() : (DY || new Object());
    DY.slim = 'true';
    DY.dyid = '7340867428391058585';
    DY.server = 'dynamicyield.com';
    DY.color = 'cg.fst.';
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
    DY.session = '9f509e2678fdabd47ad1ee6e2d039956';
    DY.section = '8765463';
    DY.secAct = false;
    DY.ignore = false;
    DY.layout = 'def';
    DY.nu = true;
    DY.adSelectors = [];
    DY.adUrlSelectors = [];
    DY.svars = [];
    DY.goalScripts = [];
    DY.audienceRules = [{
        "audience": 1832,
        "updatedAt": "2014-09-08 17:35:26",
        "rule": [{
            "condType": "Referrer",
            "conds": [{
                "id": 2955,
                "parameter": null,
                "selectMethod": "contains",
                "selectParameter": "twitter.com",
                "hitCountMethod": ">=",
                "hitCount": 1
            }
            ]
        }
        ],
        "session": 0,
        "hidden": 0
    }, {
        "audience": 1833,
        "updatedAt": "2014-09-08 17:55:00",
        "rule": [{
            "condType": "Referrer",
            "conds": [{
                "id": 2956,
                "parameter": null,
                "selectMethod": "contains",
                "selectParameter": "www.nytimes.com\/pages\/business",
                "hitCountMethod": ">=",
                "hitCount": 1
            }
            ]
        }
        ],
        "session": 1,
        "hidden": 1
    }
    ];
    DY.fpExp = [];
    DY.prd = [];
    DY.tzo = - 5;
    var dys = document.createElement("script");
    dys.setAttribute("src", "http://cdn.dynamicyield.com/scripts/6217/dy-min.js?v=3635");
    dys.setAttribute("type", "text/javascript");
    dys.setAttribute("async", "true");
} catch (exception) {}
var dyjqs = window.document.createElement('script');
dyjqs.type = 'text/javascript';
dyjqs.src = 'http://cdn.dynamicyield.com/scripts/6217/dyjq-min.js?v=3635';
dyjqs.async = true;
var dyfs = document.getElementsByTagName('script')[0];
dyfs.parentNode.insertBefore(dys, dyfs);
dyfs.parentNode.insertBefore(dyjqs, dyfs);
try {
    if (typeof DYO !== "undefined") {
        var oldSetVar = typeof DYO.setVariations !== "undefined";
        var newSetVar = typeof DYO.storeAttributeVariations !== "undefined";
        if (oldSetVar || newSetVar) {
            var setVarFunc = newSetVar ? DYO.storeAttributeVariations : DYO.setVariations;
        }
    }
} catch (e) {}
