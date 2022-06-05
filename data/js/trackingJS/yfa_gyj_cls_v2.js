var YPA_TI_CLOSE_FLAG = true;
var YPA_TARGET_SUBDOMAIN = "www.yahoo.co.jp";
var YPAcloseImg = "http://ai.yimg.jp/bdv/yahoo/images/closead.png";
var YPA_TARGET_PATH = "/";
var YPA_CLOSE_CAP_COOKIE_NAME = "RMATICW";
var YPA_COOKIE_EXPIRES_DAY = 28;
var YPAtiElement;
var YPAtilElement;
var YPAtilImgElement;
var YPAtirElement;
var YPAtirImgElement;
var YPAopenElement;
var YPAcloseElement;
var YPAtiIO;
var YPAtiADID;
var YPAcurrentTime;
var YPAdisplayFlg;
var YPArmaticwCookie;
var YPArmaticwKey;
var YPAtrhandle;
var YPAsessionId = makeSessionID();
if (YPAlinkL.indexOf("ace%")!=-1 && window.location.href.indexOf("http")!=-1) {
    YPAlinkL = "";
    YPAlinkR = "";
    YPAtarget = "";
    YPAtrhandle = true;
}
if (YPA_TI_CLOSE_FLAG) {
    tiMain();
}
function tiMain() {
    YPAtilElement = createTopImpactLTag();
    YPAtirElement = createTopImpactRTag();
    YPAtiElement = document.getElementById("topImpact");
    YPAtiElement.appendChild(YPAtilElement);
    YPAtiElement.appendChild(YPAtirElement);
    YPAtiIO = encodeBase32(YPAybx.match(/M=(\d+)/)[1]);
    YPAtiADID = encodeBase32(YPAybx.match(/A=(\d+)/)[1]);
    YPArmaticwKey = YPAtiIO;
    YPAcurrentTime = (YPAybx.match(/J=(\d+)/)[1]).substring(0, 10);
    if (!document.cookie) {
        openDisplay();
        YPAdisplayFlg = true;
    } else {
        YPArmaticwCookie = getRmaticwCookie();
        var a = YPArmaticwCookie;
        removeExpiredCookieValue();
        YPAdisplayFlg=!checkCloseCap(YPArmaticwKey);
        if (YPAdisplayFlg) {
            openDisplay();
        } else {
            closeDisplay();
            SendRedirect2Stats(makeNonDisplayBeaconURL());
        }
        if (YPArmaticwCookie != a) {
            updateCookie(YPArmaticwCookie);
        }
    }
    YPAtrhandle = false;
}
function createTopImpactLTag() {
    var a = document.createElement("div");
    a.id = "topImpactL";
    a.style.position = "absolute";
    a.style.left = "-180px";
    return a;
}
function createTopImpactRTag() {
    var a = document.createElement("div");
    a.id = "topImpactR";
    a.style.position = "absolute";
    a.style.left = "100%";
    a.style.marginLeft = "10px";
    return a;
}
function createImgTag(c, a, d) {
    var b = document.createElement("a");
    b.href = d;
    b.target = "_top";
    var e = document.createElement("img");
    e.src = "http://ai.yimg.jp/images/clear.gif";
    e.style.width = c;
    e.style.height = a;
    b.appendChild(e);
    return b;
}
function createCloseButtonTag() {
    var a = document.createElement("div");
    a.id = "closeTopImpact";
    a.style.position = "absolute";
    a.style.left = "100%";
    a.style.margin = "6px 0 0 16px";
    a.style.cursor = "pointer";
    var b = document.createElement("img");
    b.onclick = toggleDisplay;
    b.src = YPAcloseImg;
    b.alt = "広告を非表示";
    a.appendChild(b);
    return a;
}
function createOpenButtonTag() {
    var a = document.createElement("div");
    a.id = "openTopImpact";
    a.style.position = "absolute";
    a.style.left = "100%";
    a.style.margin = "7px 0 0 12px";
    a.style.fontSize = "12px";
    a.style.textDecoration = "underline";
    a.style.whiteSpace = "nowrap";
    var b = document.createElement("a");
    b.href = "javascript:void(0);";
    b.onclick = toggleDisplay;
    b.innerHTML = "広告を再表示";
    a.appendChild(b);
    return a;
}
function createButtonTag(f, c, a, e) {
    var b = document.createElement("a");
    b.id = f;
    b.href = "javascript:toggleDisplay();";
    b.style.position = "absolute";
    b.style.top = "10px";
    b.style.left = "10px";
    var d = document.createElement("img");
    d.src = YPAcloseImg;
    d.style.width = c;
    d.style.height = a;
    b.appendChild(d);
    return b;
}
function toggleDisplay() {
    removeExpiredCookieValue();
    if (YPAdisplayFlg) {
        closeDisplay();
        YPAdisplayFlg = false;
        addCookieValue(YPArmaticwKey);
        SendRedirect2Stats(makeOpenCloseBeaconURL("close"));
    } else {
        openDisplay();
        YPAdisplayFlg = true;
        removeCookieValue(YPArmaticwKey);
        SendRedirect2Stats(makeOpenCloseBeaconURL("expand"));
    }
    updateCookie(YPArmaticwCookie);
}
function closeDisplay() {
    setDisplayNone(YPAtilImgElement);
    setDisplayNone(YPAtirImgElement);
    setDisplayNone(YPAcloseElement);
    if (YPAopenElement != null) {
        YPAopenElement.style.display = "block";
    } else {
        YPAopenElement = createOpenButtonTag();
        YPAtiElement.appendChild(YPAopenElement);
    }
}
function openDisplay() {
    setDisplayNone(YPAopenElement);
    if (YPAtilImgElement != null) {
        YPAtilImgElement.style.display = "block";
    } else {
        YPAtilImgElement = createImgTag("170px", "1024px", YPAlinkL);
        YPAtilElement.appendChild(YPAtilImgElement);
        YPAtilElement.style.backgroundImage = "url(" + YPAtilImg + ")";
    }
    if (YPAtirImgElement != null) {
        YPAtirImgElement.style.display = "block";
    } else {
        YPAtirImgElement = createImgTag("170px", "1024px", YPAlinkR);
        YPAtirElement.appendChild(YPAtirImgElement);
        YPAtirElement.style.backgroundImage = "url(" + YPAtirImg + ")";
    }
    if (YPAcloseElement != null) {
        YPAcloseElement.style.display = "block";
    } else {
        YPAcloseElement = createCloseButtonTag();
        YPAtiElement.appendChild(YPAcloseElement);
    }
}
function setDisplayNone(a) {
    if (a != null) {
        a.style.display = "none";
    }
}
function checkCloseCap(a) {
    if (!document.cookie) {
        return false;
    }
    return checkValidTerm(a);
}
function checkValidTerm(b) {
    if (YPArmaticwCookie == null || YPArmaticwCookie == "") {
        return false;
    }
    var a = getExpiredTime(b);
    if (a===-1) {
        return false;
    }
    return (a > YPAcurrentTime);
}
function getExpiredTime(a) {
    if (YPArmaticwCookie == null || YPArmaticwCookie == "") {
        return - 1;
    }
    var b = YPArmaticwCookie.match(new RegExp(a + ":e=([^;]*);?"));
    return (b == null)?-1 : decodeBase32(b[1]);
}
function getRmaticwCookie() {
    var a = document.cookie.match(/RMATICW=([^;]*);?/);
    return (a == null) ? null : decodeURIComponent(a[1]);
}
function getRmaticwValueNum(a) {
    var b = a.split(";");
    return (b == null) ? 0 : b.length;
}
function addCookieValue(b) {
    var a = "";
    if (YPArmaticwCookie != null && YPArmaticwCookie != "") {
        a = YPArmaticwCookie.replace(new RegExp(b + ":e=[^;];?"), "") + ";";
    }
    a += b + ":e=" + encodeBase32(parseInt(YPAcurrentTime) + parseInt(YPAcloseCap));
    if (getRmaticwValueNum(a) > 24) {
        a = a.replace(/^[^;]*;/, "");
    }
    YPArmaticwCookie = a;
}
function removeCookieValue(a) {
    if (YPArmaticwCookie == null || YPArmaticwCookie == "") {
        return;
    }
    YPArmaticwCookie = YPArmaticwCookie.replace(new RegExp(a + ":e=[^;]*;?"), "");
}
function removeExpiredCookieValue() {
    if (YPArmaticwCookie == null || YPArmaticwCookie == "") {
        return;
    }
    var d = YPArmaticwCookie.split(";");
    var a = "";
    for (var c = 0; c < d.length; c++) {
        if (d[c] === "" || d[c] == null) {
            break;
        }
        var b = String(d[c]).match(/^([^:]*):e=/);
        if (checkValidTerm(b[1])) {
            a += d[c] + ";";
        }
    }
    a = a.slice(0, - 1);
    YPArmaticwCookie = a;
}
function updateCookie(a) {
    if (a === "" || a == null) {
        deleteCookie();
    } else {
        writeCookie(a);
    }
}
function writeCookie(a) {
    var b = removeBothEndSeparator(a);
    writeToBrowserCookie(formatRmaticw(encodeURIComponent(b), getExpiresTime()));
}
function deleteCookie() {
    writeToBrowserCookie(formatRmaticw("", "Thu, 01 Jan 1970 00:00:00 GMT"));
}
function writeToBrowserCookie(a) {
    if (a != null) {
        document.cookie = a;
    }
}
function formatRmaticw(b, a) {
    return YPA_CLOSE_CAP_COOKIE_NAME + "=" + b + ";domain=" + YPA_TARGET_SUBDOMAIN + ";path=" + YPA_TARGET_PATH + ";expires=" + a + ";";
}
function getExpiresTime() {
    return new Date(parseInt(YPAcurrentTime) * 1000 + parseInt(YPA_COOKIE_EXPIRES_DAY * 24 * 60 * 60 * 1000)).toGMTString();
}
function encodeBase32(a) {
    return parseInt(a).toString(32);
}
function decodeBase32(a) {
    return parseInt(a, 32);
}
function removeBothEndSeparator(a) {
    if (a == null || a == "") {
        return a;
    }
    var b = a;
    b = (b.charAt(0) == ";") ? b.slice(1) : b;
    b = (b.charAt(b.length - 1) == ";") ? b.slice(0, - 1) : b;
    return b;
}
function YPArdvoid() {
    return;
}
function YPAvoid() {
    return;
}
function makeNonDisplayBeaconURL() {
    var a = "action1";
    if (typeof YPAybx == "undefined" || YPAybx.indexOf("%adid%")!=-1 || YPAybx.indexOf("%matchid%")!=-1) {
        return null;
    }
    if (typeof YAHOO != "undefined" && typeof YAHOO.JP != "undefined" && typeof YAHOO.JP.rma != "undefined" && typeof YAHOO.JP.rma.outstream != "undefined") {
        a = "on_mouse";
    }
    return YPAybx + "&IVaction=" + a + "&IVvisibletime=0&IVactiontime=0&IVsid=" + YPAsessionId;
}
function makeOpenCloseBeaconURL(a) {
    if (typeof YPAybx == "undefined" || YPAybx.indexOf("%adid%")!=-1 || YPAybx.indexOf("%matchid%")!=-1) {
        return null;
    }
    return YPAybx + "&IVaction=" + a + "&IVvisibletime=0&IVactiontime=0&IVsid=" + YPAsessionId;
}
function SendRedirect2Stats(b) {
    if (!YPAtrhandle && b != null) {
        var a = new Image();
        a.onload = function() {
            YPAvoid();
        };
        a.src = b + "&r=" + YPAGenrandom(4);
    }
}
function YPAGenrandom(c) {
    var b = Math.pow(10, c);
    var a = String(Math.floor(Math.random() * b));
    while (a.length < c) {
        a = "0" + a;
    }
    return a;
}
function makeSessionID() {
    var d = 3;
    var b = String(parseInt(Math.random() * Math.pow(10, d)));
    var a =+ new Date();
    while (b.length < d) {
        b = "7" + b;
    }
    var c = b + String(a);
    return parseInt(c).toString(32);
}
