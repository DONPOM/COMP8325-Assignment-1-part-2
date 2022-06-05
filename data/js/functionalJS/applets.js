//java help page check if user completed the description field
function checkEmptyFields() {

    document.getElementById(ClientIDs.javaenabledHidden).value = document.getElementById('javaenabled').value
    document.getElementById(ClientIDs.javavendorHidden).value = document.getElementById('javavendor').value
    document.getElementById(ClientIDs.javaversionHidden).value = document.getElementById('javaversion').value
}

//check java

// *********************************************************************************
var JavaEnabled = "Unknown";
var JavaVersion = "Unknown";
var JavaVendor = "Unknown";

var agt = navigator.userAgent.toLowerCase();
var ie = (agt.indexOf("msie") != -1);
var ns = (navigator.appName.indexOf("Netscape") != -1);
var win = ((agt.indexOf("win") != -1) || (agt.indexOf("32bit") != -1));
var mac = (agt.indexOf("mac") != -1);



function detectJava() {


    var javaApplet = document.getElementById("detectjvmapplet");
    if (javaApplet != null) {
        try {
            JavaVersion = javaApplet.getJavaVersion();
            JavaVendor = javaApplet.getJavaVendor();
            JavaEnabled = "Enabled";
        }
        catch (err) {
            JavaEnabled = "Disabled";
            //JavaEnabled = "Disabled (unable to query the detect applet); " + " Message: " + err.message + "; Err. name: " + err.name + "; Err. desc.: " + err.description + "; Err number: " + (err.number & 0xFFFF);
        }
    }
    else {
        JavaEnabled = "Disabled ";
    }
}

function checkJava() {
    //if (navigator.javaEnabled()) {
    //JavaEnabled = "Enabled";
    if (ie && mac) {
        JavaVersion = "Unable to query (Mac IE)";
        JavaVendor = "Unable to query (Mac IE)";
    }
    else {
        detectJava();
    }
    //	}
    //	else {
    //		JavaEnabled = "Disabled";
    //	}

    document.getElementById('javaenabled').value = JavaEnabled;
    document.getElementById('javavendor').value = JavaVendor;
    document.getElementById('javaversion').value = JavaVersion;
}


function checkJavaGeneral() {
    if (navigator.javaEnabled()) {
        JavaEnabled = "Enabled";
        if (ie && mac) {
            JavaVersion = "Unable to query (Mac IE)";
            JavaVendor = "Unable to query (Mac IE)";
        }
        else
            detectJava();
    }
    else {
        JavaEnabled = "Disabled";
    }

    document.getElementById(ClientIDs.javaenabledHidden).value = JavaEnabled;
    document.getElementById(ClientIDs.javavendorHidden).value = JavaVendor;
    document.getElementById(ClientIDs.javaversionHidden).value = JavaVersion;
}

//end check java


function getEndUserID(appkey) {
    var endUserID = getCookie(appkey);
    if (endUserID == null) {
        endUserID = generateUniqueID();
        // Create a date far into the future
        var today = new Date();
        today.setTime(today.getTime() + (86400000 * 18000));
        setCookie(appkey, endUserID, today);
    }
    return endUserID;
}

// Generates a new unique end-user id on the fly //
function generateUniqueID() {
    var strId = "";
    for (i = 0; i < 5; i++) {
        strId += String.fromCharCode(getRandomValue(90, 65));
    }
    var today = new Date();
    strId += today.getTime()
    return strId;
}

// Generates a random number between 'min' and 'max' //
function getRandomValue(max, min) {
    var span = max - min;
    var d = span * Math.random();
    return d + min;
}

// Sets cookie values. Expiration date is optional // 
function setCookie(name, value, expire) {
    document.cookie = name + "=" + escape(value) + ((expire == null) ? "" : ("; expires=" + expire.toGMTString()));
}

// Returns the value of the cookie with the given name //
// this fixes an issue with the old method, ambiguous values 
// with this test document.cookie.indexOf( name + "=" );
function getCookie(check_name) {
    // first we'll split this cookie up into name/value pairs
    // note: document.cookie only returns name=value, not the other components
    var a_all_cookies = document.cookie.split(';');
    var a_temp_cookie = '';
    var cookie_name = '';
    var cookie_value = '';
    var b_cookie_found = false; // set boolean t/f default f

    for (i = 0; i < a_all_cookies.length; i++) {
        // now we'll split apart each name=value pair
        a_temp_cookie = a_all_cookies[i].split('=');


        // and trim left/right whitespace while we're at it
        cookie_name = a_temp_cookie[0].replace(/^\s+|\s+$/g, '');

        // if the extracted name matches passed check_name
        if (cookie_name == check_name) {
            b_cookie_found = true;
            // we need to handle case where cookie has no value but exists (no = sign, that is):
            if (a_temp_cookie.length > 1) {
                cookie_value = unescape(a_temp_cookie[1].replace(/^\s+|\s+$/g, ''));
            }
            // note that in cases where cookie is initialized but no value, null is returned
            return cookie_value;
            break;
        }
        a_temp_cookie = null;
        cookie_name = '';
    }
    if (!b_cookie_found) {
        return null;
    }
}

var BrowserDetect = {
    init: function() {
        this.browser = this.searchString(this.dataBrowser) || "An unknown browser";
        this.version = this.searchVersion(navigator.userAgent)
			|| this.searchVersion(navigator.appVersion)
			|| "an unknown version";
        this.OS = this.searchString(this.dataOS) || "an unknown OS";
    },
    searchString: function(data) {
        for (var i = 0; i < data.length; i++) {
            var dataString = data[i].string;
            var dataProp = data[i].prop;
            this.versionSearchString = data[i].versionSearch || data[i].identity;
            if (dataString) {
                if (dataString.indexOf(data[i].subString) != -1)
                    return data[i].identity;
            }
            else if (dataProp)
                return data[i].identity;
        }
    },
    searchVersion: function(dataString) {
        var index = dataString.indexOf(this.versionSearchString);
        if (index == -1) return;
        return parseFloat(dataString.substring(index + this.versionSearchString.length + 1));
    },
    dataBrowser: [
		{ string: navigator.userAgent,
		    subString: "OmniWeb",
		    versionSearch: "OmniWeb/",
		    identity: "OmniWeb"
		},
		{
		    string: navigator.vendor,
		    subString: "Apple",
		    identity: "Safari"
		},
		{
		    prop: window.opera,
		    identity: "Opera"
		},
		{
		    string: navigator.vendor,
		    subString: "iCab",
		    identity: "iCab"
		},
		{
		    string: navigator.vendor,
		    subString: "KDE",
		    identity: "Konqueror"
		},
		{
		    string: navigator.userAgent,
		    subString: "Firefox",
		    identity: "Firefox"
		},
		{
		    string: navigator.vendor,
		    subString: "Camino",
		    identity: "Camino"
		},
		{		// for newer Netscapes (6+)
		    string: navigator.userAgent,
		    subString: "Netscape",
		    identity: "Netscape"
		},
		{
		    string: navigator.userAgent,
		    subString: "MSIE",
		    identity: "Explorer",
		    versionSearch: "MSIE"
		},
		{
		    string: navigator.vendor,
		    subString: "Google",
		    identity: "Google"
		},
		{
		    string: navigator.userAgent,
		    subString: "Gecko",
		    identity: "Mozilla",
		    versionSearch: "rv"
		},
		{ 		// for older Netscapes (4-)
		    string: navigator.userAgent,
		    subString: "Mozilla",
		    identity: "Netscape",
		    versionSearch: "Mozilla"
		}
	],
    dataOS: [
		{
		    string: navigator.platform,
		    subString: "Win",
		    identity: "Windows"
		},
		{
		    string: navigator.platform,
		    subString: "Mac",
		    identity: "Mac"
		},
		{
		    string: navigator.platform,
		    subString: "Linux",
		    identity: "Linux"
		}
	]

};
//BrowserDetect.init();
//end old website scripts

function setLanguage(language, applet) {
    var applet = document.getElementById(applet);
    applet.setLanguage(language);
    //alert(language);
}
