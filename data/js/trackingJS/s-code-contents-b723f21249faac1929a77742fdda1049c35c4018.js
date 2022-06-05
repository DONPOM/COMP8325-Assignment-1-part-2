/*! adbadobenonacdc 13-11-2014 09:54:03 */
var s_adbadobenonacdc, s_adobe;
!function(a) {
    "use strict";
    var b = window, c = "s_adbadobenonacdc", d = new AppMeasurement;
    if (b[c] = d, b.s_adobe = d, d.account = a._getAdobeAnalyticsAccount(c), "staging" !== a._getDTMEnvironment()) {
        var e = b.location.hostname.toLowerCase();
        - 1 !== e.indexOf("adobe.com") && (d.account =- 1 !== e.indexOf(".dev.adobe.")||-1 !== e.indexOf(".corp.adobe")||-1 !== e.indexOf(".dev0")||-1 !== e.indexOf(".qa0")||-1 !== e.indexOf(".pr0")||-1 !== e.indexOf("helpx.qa.")||-1 !== e.indexOf("day.adobe.com")||-1 !== e.indexOf("stage.")||-1 !== e.indexOf("stage2.")||-1 !== e.indexOf("staging.") ? "adbadobenonacdcqa" : "adbadobenonacdcprod")
    }
    d.charSet = "UTF-8", d.trackingServer = "stats.adobe.com", d.trackingServerSecure = "sstats.adobe.com", d.cookieDomainPeriods = a._getDomainPeriods(), d.fpCookieDomainPeriods = a._getDomainPeriods(), d.trackInlineStats=!0, d.trackDownloadLinks=!0, d.trackExternalLinks=!0, d.linkLeaveQueryString=!1, d.linkTrackEvents = "None", d.linkTrackVars = "None", d.linkDownloadFileTypes = ["adpp", "air", "apk", "avi", "bin", "cptx", "css", "csv", "dmg", "doc", "docx", "eps", "exe", "flv", "hqx", "jar", "jpg", "js", "m4v", "mov", "mp3", "mpg", "msi", "mxp", "pdf", "png", "ppt", "pptx", "rar", "svg", "swc", "tab", "tbz2", "txt", "vsd", "vxd", "wav", "wma", "wmv", "xls", "xlsx", "xml", "zip", "zxp"].join(","), d.linkExternalFilters = "", d.linkInternalFilters = ["acrobat.com", "acrobatclub.com", "acrobatusers.com", "adobe-createnow.jp", "adobe-creativity.com", "adobe-designforimpact.com", "adobe-digital-marketing.com", "adobe-edu.ru", "adobe-education.com", "adobe-eseminars.de", "adobe-events-france.com", "adobe-html5.jp", "adobe-max.com", "adobe-mobile.co.uk", "adobe-mobile.com", "adobe-mobile.de", "adobe-mobile.fr", "adobe-newsroom.de", "adobe-onlineservices.com", "adobe-pepe.jp", "adobe-promo.jp", "adobe-ria.jp", "adobe-solutions.de", "adobe-target.com", "adobe.", "adobeacrobat.com", "adobeaemcloud.com", "adobeawards.com", "adobecc.com", "adobeceea.com", "adobecoopmdf.com", "adobecqcloud.com", "adobecreate.co.uk", "adobecreatenow-apac.com", "adobecreativecloud.com", "adobecreativesuite.it", "adobecs2.com", "adobecs3.com", "adobedigitalmarketing.co.uk", "adobedms.pl", "adobeexchange.com", "adobeeventsonline.com", "adobeformscentral.com", "adobeforums.com", "adobegeneration.anz.com", "adobegeneration.co.uk", "adobegeneration.com", "adobegov.com", "adobegunlugu.com", "adobeindia.com", "adobekb.com", "adobeknowhow.com", "adobeku.com", "adobeleanprint.com", "adobemarketing.co.uk", "adobemarketplace.com", "adobeme.com", "adobepartnernews.com", "adoberatingreviews.webqamapps.com", "adobestudentlicensing.at", "adobestudentlicensing.ch", "adobestudentlicensing.de", "adobesunbreak.com", "adobesystems.", "adobetraining.pro", "appspot.com", "behance.net", "businesscatalyst.com", "canaladobe.com", "channelconduit.com", "cmoconnect.com", "douwriteright.com", "dpssummit.com", "echosign.com", "flex.org", "gocreate.pro", "hellodigitals.it", "jaknapdf.cz", "javascript:", "lightroomcomp.jp", "loveacrobat.co.za", "mailto:", "macromedia.", "mediacenter-adobe.com", "openscreenproject.org", "photoshop.com", "rhapsodyconcept.com.sg", "runaware.com", "scene7.com", "tel:", "worldsecuresystems.com", "../"].join(",");
    var f = {};
    d.usePlugins=!0, d.doPlugins = function(c) {
        var d = 0, e=!1;
        if ("file:" == document.location.protocol, b.SiteCatalyst) {
            b.SiteCatalyst.productEvents && (c.events = a._apl(c.events, b.SiteCatalyst.productEvents, ",", 2));
            var g = "";
            b.SiteCatalyst.productKey ? g = b.SiteCatalyst.productKey : b.SiteCatalyst.sitecatalystProductKey && (g = b.SiteCatalyst.sitecatalystProductKey), "string" == typeof g && g && ( - 1 !== g.indexOf("/Applications/") && (g = g.replace("/Applications/", "")), g = ["", g, "", "", "", ""].join(";"), c.products = a._apl(c.products, g, ",", 2))
        }
        c.pageName&&-1 != c.pageName.indexOf(":products:") && c.products && (c.prop2 = c.products.split(";")[1]), c.prop14 && (c.prop14 = c.prop14.toLowerCase()), d = a._getVisitStart("s_vs"), d && 1 == d && (e = "firstpage"), c.clickPast(e, "event19", "event20"), c.prop6 && (c.eVar3 = c.prop6, c.events = a._apl(c.events, "event1", ",", 2), !c.prop7 || "0" != c.prop7 && "zero" != c.prop7 || (c.prop7 = "zero", c.events = a._apl(c.events, "event2", ",", 2)));
        var h = a._getValOnce(c.eVar3, "s_stv", 0);
        if ("" === h) {
            var i, j = c.events ? c.events.split(","): [], k = [];
            for (i = 0; i < j.length; i++)
                "event1" != j[i] && "event2" != j[i] && k.push(j[i]);
            c.events = k.join(",")
        }
        if (c.events&&-1 !== c.events.indexOf("prodView") && (c.events = a._apl(c.events, "event3", ",", 2)), c.purchaseID && (c.eVar27 = c.purchaseID), c.eVar1 && (!c.products || c.products && c.products.indexOf(";productmerch")>-1 || "true" == c.newProduct) && (!f.onemerch || "" !== c.linkType && c.linkTrackVars.indexOf("eVar1")>-1)) {
            f.onemerch=!0, c.productNum = a._readCookie("productnum") ? parseInt(a._readCookie("productnum")) + 1 : 1, c.products = ";productmerch" + c.productNum;
            var l = new Date;
            l.setTime(l.getTime() + 2592e6), a._setCookie("productnum", c.productNum, l), c.linkTrackVars = a._apl(c.linkTrackVars, "events,products", ",", 2), c.linkTrackEvents = a._apl(c.linkTrackEvents, "event13", ",", 2), c.events = a._apl(c.events, "event13", ",", 2)
        }
        a._readCookie("productnum") && c.events.indexOf("purchase")>-1 && a._setCookie("productnum", "0", 0), c.events && c.events.indexOf("scAdd")>-1 && (c.linkTrackVars = a._apl(c.linkTrackVars, "eVar23", ",", 2), c.prop12 && (c.eVar23 = c.prop12)), c.products&&!c.events && (c.products = ""), "" !== a._readCookie("aam_tnt") && (c.list1 = a._unescape(a._readCookie("aam_tnt"))), b.mboxVersion && "function" == typeof b.mboxLoadSCPlugin && (b.mboxLoadSCPlugin(c), c.tnt = c._plugin_trackTNT())
    }, a._adobeAnalytics_plugin_clickPast(d), a._adobeAnalytics_plugin_combinedCookies(d), a._adobeAnalytics_plugin_socialAuthors(d), a._adobeAnalytics_plugin_getTrackingServer(d), a._adobeAnalytics_plugin_trackTNT(d), a._adobeAnalytics_plugin_demandbaseDataConnector(d, {
        key: "e4086fa3ea9d74ac2aae2719a0e5285dc7075d7b",
        var_name: "s_demandbase_v2.2",
        contextName: "s_dmdbase",
        dimensionArray: [{
            id: "demandbase_sid",
            max_size: 10
        }, {
            id: "company_name",
            max_size: 40
        }, {
            id: "industry",
            max_size: 40
        }, {
            id: "sub_industry",
            max_size: 40
        }, {
            id: "employee_range",
            max_size: 30
        }, {
            id: "revenue_range",
            max_size: 10
        }, {
            id: "audience",
            max_size: 30
        }, {
            id: "audience_segment",
            max_size: 30
        }
        ],
        contextNameCustom: "s_dmdbase_custom",
        dimensionArrayCustom: [{
            id: "fortune_1000",
            max_size: 5
        }, {
            id: "state",
            max_size: 4
        }, {
            id: "zip",
            max_size: 12
        }, {
            id: "country",
            max_size: 4
        }, {
            id: "b2b",
            max_size: 5
        }, {
            id: "b2c",
            max_size: 5
        }, {
            id: "watch_list_top100",
            max_size: 20
        }, {
            id: "tech_insights",
            max_size: 30
        }
        ]
    }), a.getVar("isSite_AdobeHelp") || a.getVar("isSite_AdobeHelpX") || d._plugin_demandbaseDataConnector(), a._adobeAnalytics_plugin_demandbase(d), a.getVar("isSite_AdobeHelp") || a.getVar("isSite_AdobeHelpX") || d._plugin_demandbase({
        key: "e4086fa3ea9d74ac2aae2719a0e5285dc7075d7b",
        data_mapping: {
            sid: "eVar56",
            company_name: "eVar50",
            industry_complete: "eVar51",
            employee_range: "eVar52",
            revenue_range: "eVar53"
        },
        var_name: "s_adobe_dmdbase_v_1",
        link_name: "Genesis: Demandbase event"
    }), a._adobeAnalytics_plugin_trackSynchronous(d), a.getVar("useModule_media") && (d.enableVideoTracking=!0, d.loadModule("Media"), d.Media.autoTrack=!1, d.Media.trackVars = "events,prop8,eVar9,eVar10,eVar11", d.Media.trackEvents = "event4,event5,event6,event7,event8,event9,event10", d.Media.trackMilestones = "25,50,75", d.Media.playerName =- 1 !== b.location.hostname.toLowerCase().indexOf("tv.adobe.com") ? "AdobeTV HTML5 Player" : "Microsite HTML5 Player", d.Media.segmentByMilestones=!0, d.Media.trackUsingContextData=!0, d.Media.contextDataMapping = {
        "a.media.name": "eVar9,prop8",
        "a.media.segment": "eVar10",
        "a.contentType": "eVar11",
        "a.media.timePlayed": "event4",
        "a.media.view": "event5",
        "a.media.segmentView": "event6",
        "a.media.complete": "event10",
        "a.media.milestones": {
            25: "event7",
            50: "event8",
            75: "event9"
        }
    }), d.loadModule("Integrate"), d.Integrate.onLoad = function(a) {
        a.socialAuthors()
    }, a.getVar("useTool_audienceManager") && a._runWhenResolved(function() {
        var a, c = {
            partner: "adobe",
            uuidCookie: {
                name: "aam_uuid",
                days: 30
            },
            declaredId: {}
        };
        b.adobeIMS._profile && (c.declaredId.dpuuid = b.adobeIMS._profile.userId.split("@")[0], c.declaredId.dpid = "813"), a = b.DIL.create(c), b.DIL.modules.siteCatalyst.init(d, a, {
            names: ["pageName", "channel", "campaign", "products", "events", "pe", "referrer", "server", "purchaseID", "zip", "state"],
            iteratedNames: [{
                name: "eVar",
                maxIndex: 75
            }, {
                name: "prop",
                maxIndex: 75
            }, {
                name: "pev",
                maxIndex: 3
            }, {
                name: "hier",
                maxIndex: 4
            }
            ]
        })
    }, [function() {
        return b.DIL && "undefined" != typeof b.adobeIMS && "undefined" != typeof b.adobeIMS._profile
    }
    ], {
        timeout: 4e3,
        interval: 100
    })
}(_satellite);
