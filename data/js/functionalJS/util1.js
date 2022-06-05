/*util.js*/

function isCSAppletVisible() {
    var isVis = true;
    var hide = getCookieNew("hideAppletsCS");
    if (hide == null || hide == "") {
        isVis = false;
    }
    else {
        isVis = true;
    }
    //return isVis;
    return true;
}

function isAppletVisible() {
    var isVis = true;
    var hide = getCookieNew("hideApplets");
    if (hide == null || hide == "") {
        isVis = false;
    }
    else {
        isVis = true;
    }
    return isVis;
}

function isAppletVisibleFP() {
    var isVis = true;
    var hide = getCookieNew("hideApplets");
    if (hide == null || hide == "") {
        isVis = false;
    }
    else {
        isVis = true;
    }
    return isVis;
    //return true;
}


function hideApplets() {

    var hide = getCookieNew("hideApplets");
    if (hide == null || hide == "") {

        var ql = document.getElementById('qlApplet');
        if (ql != null) {
            ql.style.display = 'none';
        }
        var fc = document.getElementById('fcApplet');
        if (fc != null) {
            fc.style.display = 'none';
        }

        //csApplet
        var cs = document.getElementById('csApplet');
        if (cs != null) {
            cs.style.display = 'none';
        }
        //quoteApplet
        var qa = document.getElementById('quoteApplet');
        if (qa != null) {
            qa.style.display = 'none';
        }

        //thinChart
        var tc = document.getElementById('thinChart');
        if (tc != null) {
            tc.style.display = 'none';
        }

        //newsApplet
        var na = document.getElementById('newsApplet');
        if (na != null) {
            na.style.display = 'none';
        }

        //marketTicker
        var mt = document.getElementById('marketTicker');
        if (mt != null) {
            mt.style.display = 'none';
        }

        //front page
        var fp = document.getElementById('divApplet');
        if (fp != null) {
            fp.style.display = 'none';
        }

        //netstation
        var ns = document.getElementById('netstation');
        if (ns != null) {
            ns.style.display = 'none';
        }


    }
}

function isBelowIEX(ieVersion) {

    var version = ieVersion || 9;

    if ($.browser.msie && parseInt($.browser.version, 10) < version) {
        return true;
    }
    return false;
}

// Returns the value of the cookie with the given name //
// this fixes an issue with the old method, ambiguous values 
// with this test document.cookie.indexOf( name + "=" );
function getCookieNew(check_name) {
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

// Sets cookie values. Expiration date is optional // 
function setCookieNew(name, value, expire) {
    document.cookie = name + "=" + escape(value) + ((expire == null) ? "" : ("; expires=" + expire.toGMTString()));
}



    function hideModalPopupViaClient() {
        var modalPopupBehavior = $find('programmaticModalPopupBehavior');
        modalPopupBehavior.hide();
        setCookieNew("hideApplets", "false");
        showApplets();
    }

	function hideModalPopupViaClientCS() {
        var modalPopupBehavior = $find('programmaticModalPopupBehavior');
        modalPopupBehavior.hide();
        setCookieNew("hideAppletsCS", "false", null);
        showApplets();
    }


    function showApplets() {
        var quotelist = document.getElementById('qlApplet');
        if (quotelist != null) {
            quotelist.style.display = 'block';
        }


        var financeChart = document.getElementById('fcApplet');
        if (financeChart != null) {
            financeChart.style.display = 'block';
        }

        var cs = document.getElementById('csApplet');
        if (cs != null) {
            cs.style.display = 'block';
        }

        //quoteApplet
        var qa = document.getElementById('quoteApplet');
        if (qa != null) {
            qa.style.display = 'block';
        }

        //thinChart
        var tc = document.getElementById('thinChart');
        if (tc != null) {
            tc.style.display = 'block';
        }


        //newsApplet
        var na = document.getElementById('newsApplet');
        if (na != null) {
            na.style.display = 'block';
        }

        //marketTicker
        var mt = document.getElementById('marketTicker');
        if (mt != null) {
            mt.style.display = 'block';
        }

        //front page
        var fp = document.getElementById('divApplet');
        if (fp != null) {
            fp.style.display = 'block';
        }

        //netstation
        var ns = document.getElementById('netstation');
        if (ns != null) {
            ns.style.display = 'block';
        }
    }

    function show(tab) {
        var intro = document.getElementById("intro");
        var features = document.getElementById("features");
        var screenshots = document.getElementById("screenshots");
        var showIntro = tab == 'intro';
        var showFeatures = tab == 'features';
        var showScreenshots = tab == 'screenshots';

        if (showIntro) {
            intro.style.display = "block";
        }
        else {
            intro.style.display = "none";
        }

        if (showFeatures) {
            features.style.display = "block";
        }
        else {
            features.style.display = "none";
        }

        if (showScreenshots) {
            screenshots.style.display = "block";
        }
        else {
            screenshots.style.display = "none";
        }
    }

    function showSlideShow(tab) {
        var iphone = document.getElementById("wrapperiPhone");
        var ipad = document.getElementById("wrapperiPad");
        var android = document.getElementById("wrapperAndroid");
        var blackberry = document.getElementById("wrapperBlackberry");
        var showIPhone = tab == 'iphone';
        var showIPad = tab == 'ipad';
        var showAndroid = tab == 'android';
        var showBlackberry = tab == 'blackberry';

        var logoIPad = document.getElementById('iPadLogo');
        var logoIPhone = document.getElementById('iPhoneLogo');
        var logoAndroid = document.getElementById('androidLogo');
        var logoBlackberry = document.getElementById('blackberryLogo');

      

        if (showIPhone) {
            iphone.style.display = "block";
            ipad.style.display = "none";
            android.style.display = "none";
            blackberry.style.display = "none";
            logoIPhone.src = "/App_Themes/Default/iphone/menupopup/logo_iPhone.png";
            logoIPad.src = "/App_Themes/Default/iphone/menupopup/logo_iPad_unsel.png";
            logoAndroid.src = "/App_Themes/Default/iphone/menupopup/logo_Android_unsel.png";
            logoBlackberry.src = "/App_Themes/Default/iphone/menupopup/logo_Blackberry_unsel.png";
        }
        else if (showIPad)
        {
            iphone.style.display = "none";
            ipad.style.display = "block";
            android.style.display = "none";
            blackberry.style.display = "none";
            logoIPhone.src = "/App_Themes/Default/iphone/menupopup/logo_iPhone_unsel.png";
            logoIPad.src = "/App_Themes/Default/iphone/menupopup/logo_iPad.png";
            logoAndroid.src = "/App_Themes/Default/iphone/menupopup/logo_Android_unsel.png";
            logoBlackberry.src = "/App_Themes/Default/iphone/menupopup/logo_Blackberry_unsel.png";
        }
        else if (showAndroid) {
            iphone.style.display = "none";
            ipad.style.display = "none";
            android.style.display = "block";
            blackberry.style.display = "none";
            logoIPhone.src = "/App_Themes/Default/iphone/menupopup/logo_iPhone_unsel.png";
            logoIPad.src = "/App_Themes/Default/iphone/menupopup/logo_iPad_unsel.png";
            logoAndroid.src = "/App_Themes/Default/iphone/menupopup/logo_Android.png";
            logoBlackberry.src = "/App_Themes/Default/iphone/menupopup/logo_Blackberry_unsel.png";
        }
        else if (showBlackberry) {
            iphone.style.display = "none";
            ipad.style.display = "none";
            android.style.display = "none";
            blackberry.style.display = "block";
            logoIPhone.src = "/App_Themes/Default/iphone/menupopup/logo_iPhone_unsel.png";
            logoIPad.src = "/App_Themes/Default/iphone/menupopup/logo_iPad_unsel.png";
            logoAndroid.src = "/App_Themes/Default/iphone/menupopup/logo_Android_unsel.png";
            logoBlackberry.src = "/App_Themes/Default/iphone/menupopup/logo_Blackberry.png";
        }
    }

