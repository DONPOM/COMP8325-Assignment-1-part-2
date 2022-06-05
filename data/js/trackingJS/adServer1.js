var ebPtcl = "http://", ebBigS = "ds.serving-sys.com/BurstingCachedScripts/", ebResourcePath = "ds.serving-sys.com/BurstingRes//";
ebRand = ("" + Math.random()).substr(2);
ebBigS = ebPtcl + ebBigS;
ebResourcePath = ebPtcl + ebResourcePath;
var dmg = "undefined" != typeof gEBMainWindow && void 0 !== gEBMainWindow.EBP && void 0 !== gEBMainWindow.EBP.downloadMgr ? gEBMainWindow.EBP.downloadMgr : !1;
var placementId = 11594557, asc = dmg && void 0 !== dmg.ebSync && void 0 !== dmg.ebSync.async && void 0 !== dmg.ebSync.placementIdObj[placementId] && dmg.ebSync.placementIdObj[placementId].async;
gEbUT = "";
gEbUT = "tp_PlacementID%3D11594557%24%24tp_AdID%3D24202311%24%24";
function ebTX(e) {
    try {
        for (var t = {}, r = unescape(gEbUT).split("$$"), n = 0; n < r.length; n++) {
            var i = r[n];
            if (i) {
                var a = r[n].indexOf("="), o = r[n].substr(0, a), s = r[n].substr(a + 1);
                t[o] = escape(s)
            }
        }
        for (var u in t)
            e = e.replace(RegExp("\\[%" + u + "%\\]", "ig"), t[u]);
        return e
    } catch (g) {}
}
function ebTokens(e) {
    return "function" == typeof ebTX && (e = ebTX(e)), e.replace(/\[ebRandom\]/gi, ebRand).replace(/\[timestamp\]/gi, ebRand).replace(/\[%tp_adid%\]/gi, 24202311).replace(/\[%tp_flightid%\]/gi, 11594557).replace(/\[%tp_campaignid%\]/gi, 468176).replace(/\[%random%\]/gi, ebRand)
}
function ebReport(e) {
    asc ? dmg.reportImg(e) : document.write("<IMG SRC=" + e + " width=0 height=0 style='position:absolute;left:0px;top:0px;'></IMG>")
}
for (var a = [], i = 0; i < a.length; i++)
    ebReport(ebTokens(a[i]));
ebO = {
    pi: 0,
    sms: "ds.serving-sys.com/BurstingScript/",
    bs: "bs.serving-sys.com",
    p: "",
    tn: "Banner",
    bt: 4,
    bv: 7.100000,
    plt: 23,
    ut: gEbUT,
    dt: 0,
    bc: 3,
    IABMS: 50,
    IABMD: 1,
    IABBV: 0,
    IABEV: 0,
    AcCP: 0,
    irt: 90000,
    sid: "360081535343892084",
    d: 0,
    h5mi: {
        "EBVPAID": "/BurstingScript/EBVPAID.js",
        "adkitversion": "1_0_0_3"
    },
    fvp: "Res/",
    dlm: 1,
    usercookie: "u2=432d83a7-d317-49b0-83fa-81de62055bbc",
    oo: 0,
    ncu: "http://a.tribalfusion.com/h.click/atmQCNUV352Uqrnd6pXTeM2drZaPc7C2mBFmWItVWF8YbrcYFYh0qZanSr3ZdWFB3TdY4nrBxQFMsYarq3TFh4TUPoaBBYrU6UHBXnmfInVYwoWfA3qF73WIo4ArGnFbLYcbP1snV0VZbvparV3bZb2WFbZbUAf5REMQQVZbqSHUx1dfqTPrN4sYVYUULVA6q5mQbQPMKPaQEncIJGR/",
    nClk: 1,
    aiE: {},
    ffs: {
        "CFF_CloseAdPart": false,
        "CFF_DynamicExpansion": true,
        "CFF_RemoveUivz": false,
        "DelayedImpressionLogging": true,
        "EdsSkipDuplicates": false,
        "EnableExtendedLogging": true,
        "EnableNewAOasEven": false,
        "EnableNewFFFormat": true,
        "VersaTagPreServing": false,
        "sf": true
    },
    dex: 0,
    ta: "-1",
    dg: "5211304",
    sdg: "5211304",
    mvfd: "0",
    extensionHooks: [],
    rnd: ebRand,
    w: 728,
    h: 90,
    reportFrequency: 0
};
ebO.adConfig = AdConfig_24202311 = {
    adId: 24202311,
    placementId: 11594557,
    campaignId: 468176,
    defaultImagePath: "Site-12898/Type-0/7d672a87-fa7b-4bc0-8ae8-ae0fcb50a8e3.gif",
    templateName: "Banner",
    clickTrackingUrls: [],
    defaultFlashPath: "Site-12898/Type-2/cb20d77a-f15b-426c-9ec8-00c6f1dc9ad0.swf",
    defaultFlashWidth: 728,
    defaultFlashHeight: 90,
    richFlashPath: "Site-12898/Type-2/fe8497f1-4326-4d53-b9e3-979a60530dde.swf",
    richFlashWidth: 728,
    richFlashHeight: 90,
    richFlashWeight: 81994,
    playOnCommand: 0,
    preloader: 0,
    minimalflashVer: 11.2,
    AcIconPosition: 0,
    AcIncludeMarker: 0,
    isSVSVP : false,
    UiVz : false,
    assets: {},
    interactions: {},
    pckAssets: {},
    DynamicContentResources: {}
};
AdConfig_24202311._addInts = function () {
    for (var i = 0; i < arguments.length; i++) {
        var int = arguments[i];
        var name = int[0];
        this.interactions[name] = {
            jumpUrl: int[1],
            xPos: int[2],
            yPos: int[3],
            width: int[4],
            height: int[5],
            showAddressBar: int[6],
            showMenuBar: int[7],
            fClose: int[8],
            target: int[9],
            fClick: int[10],
            type: int[11],
            nInitiated: int[12],
            agencyURL: int[13],
            networkURL: int[14] 
        }
    }
};
AdConfig_24202311._addInts( ["_eyeblaster", "http://www.vodafone.com.au/personal/android/samsung-galaxy-note4?cid=cbu:post:mob:new:premdis:ret:projectt:exp", 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, "", ""] );;
ebO.scv = "_2_33_3_0";
ebO.sdkv = "_2_15_0_0";
ebO.rpv = "_2_8_2_0";
ebO.html5v = "_2_26_1_0";
ebO.imgv = "_2_5_3_4";
ebO.vfp = "_1_2_0_0";
ebOArr = window.ebOArr || [];
ebOArr[ebOArr.length] = ebO;
function isGlobalDefined(e, t) {
    var a = e in t;
    return "undefined" == a && (a=!1), a
}
ebIfrm = ebO.ifrm > 0;
ebO.phid = asc ? dmg.ebSync.divId : "ebDiv" + ebRand;
asc || document.write("<div id='" + ebO.phid + "' dir='ltr'></div>");
var templateName = ebO.tn;
1 != ebO.IABBV && 1 != ebO.IABEV || "StdBanner" != ebO.tn || (templateName += "Ex");
var ebSrc = ebO.ebSrc = (ebO.adConfig && ebO.adConfig.rpBigScript && ebPtcl + ebO.adConfig.rpBigScript) || ebBigS + "/Ad" + ebO.scv + "/eb" + templateName + ("undefined" != typeof ebAdCS && ebAdCS ? "_api" : "") + ".js";
asc ? dmg.loadBigScript(ebO.phid, ebSrc, ebO.tn, document) : document.write("<scr" + "ipt src=" + ebSrc + "></scr" + "ipt>");
var ebAdID = 24202311, ebPli = 11594557, ebTN = "Banner", ebDSGID = 5211304; /*8*/
