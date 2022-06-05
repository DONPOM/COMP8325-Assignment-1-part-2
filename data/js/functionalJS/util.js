var Netdania = Netdania || {};

Netdania.Util = Netdania.Util || {};


Netdania.Util.dateFormat = function () {
    var token = /d{1,4}|m{1,4}|yy(?:yy)?|([HhMsTt])\1?|[LloSZ]|"[^"]*"|'[^']*'/g,
        timezone = /\b(?:[PMCEA][SDP]T|(?:Pacific|Mountain|Central|Eastern|Atlantic) (?:Standard|Daylight|Prevailing) Time|(?:GMT|UTC)(?:[\-+]\d{4})?)\b/g,
        timezoneClip = /[^\-+\dA-Z]/g,
        pad = function (val, len) {
            val = String(val);
            len = len || 2;
            while (val.length < len) {
                val = "0" + val;
            }
            return val;
        };

    // Regexes and supporting functions are cached through closure
    return function (date, mask, utc) {
        var dF = Netdania.Util.dateFormat;

        // You can't provide utc if you skip other args (use the "UTC:" mask prefix)
        if (arguments.length == 1 && (typeof date == "string" || date instanceof String) && !/\d/.test(date)) {
            mask = date;
            date = undefined;
        }

        // Passing date through Date applies Date.parse, if necessary
        date = date ? new Date(date) : new Date();
        if (isNaN(date)) {
            throw new SyntaxError("invalid date");
        }

        mask = String(dF.masks[mask] || mask || dF.masks["default"]);

        // Allow setting the utc argument via the mask
        if (mask.slice(0, 4) == "UTC:") {
            mask = mask.slice(4);
            utc = true;
        }

        var _ = utc ? "getUTC" : "get",
            d = date[_ + "Date"](),
            D = date[_ + "Day"](),
            m = date[_ + "Month"](),
            y = date[_ + "FullYear"](),
            H = date[_ + "Hours"](),
            M = date[_ + "Minutes"](),
            s = date[_ + "Seconds"](),
            L = date[_ + "Milliseconds"](),
            o = utc ? 0 : date.getTimezoneOffset(),
            flags = {
                d:d,
                dd:pad(d),
                ddd:dF.i18n.dayNames[D],
                dddd:dF.i18n.dayNames[D + 7],
                m:m + 1,
                mm:pad(m + 1),
                mmm:dF.i18n.monthNames[m],
                mmmm:dF.i18n.monthNames[m + 12],
                yy:String(y).slice(2),
                yyyy:y,
                h:H % 12 || 12,
                hh:pad(H % 12 || 12),
                H:H,
                HH:pad(H),
                M:M,
                MM:pad(M),
                s:s,
                ss:pad(s),
                l:pad(L, 3),
                L:pad(L > 99 ? Math.round(L / 10) : L),
                t:H < 12 ? "a" : "p",
                tt:H < 12 ? "am" : "pm",
                T:H < 12 ? "A" : "P",
                TT:H < 12 ? "AM" : "PM",
                Z:utc ? "UTC" : (String(date).match(timezone) || [""]).pop().replace(timezoneClip, ""),
                o:(o > 0 ? "-" : "+") + pad(Math.floor(Math.abs(o) / 60) * 100 + Math.abs(o) % 60, 4),
                S:["th", "st", "nd", "rd"][d % 10 > 3 ? 0 : (d % 100 - d % 10 != 10) * d % 10]
            };

        return mask.replace(token, function ($0) {
            return $0 in flags ? flags[$0] : $0.slice(1, $0.length - 1);
        });
    };
}();

// Some common format strings
Netdania.Util.dateFormat.masks = {
    "default":"ddd mmm dd yyyy HH:MM:ss",
    shortDate:"m/d/yy",
    mediumDate:"mmm d, yyyy",
    longDate:"mmmm d, yyyy",
    fullDate:"dddd, mmmm d, yyyy",
    shortTime:"h:MM TT",
    mediumTime:"h:MM:ss TT",
    longTime:"h:MM:ss TT Z",
    isoDate:"yyyy-mm-dd",
    isoTime:"HH:MM:ss",
    isoDateTime:"yyyy-mm-dd'T'HH:MM:ss",
    isoUtcDateTime:"UTC:yyyy-mm-dd'T'HH:MM:ss'Z'"
};

// Internationalization strings
Netdania.Util.dateFormat.i18n = {
    dayNames:[
        "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat",
        "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
    ],
    monthNames:[
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        //"Januar", "Februar", "Marts", "April", "Maj", "Juni", "Juli", "August", "September", "Oktober", "November", "December"
        "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"

    ]
};

Netdania.Util.isInArray = function (val, arr) {
    var isInArr = false;
    for (i = 0; i < arr.length; i++) {
        if (arr[i] == val) {
            isInArr = true;
            break;
        }
    }
    return isInArr;
}

Netdania.Util.removeTableRow = function (jQtable) {
    jQtable.each(function () {
        if ($('tbody', this).length > 0) {
            $('tbody tr:last', this).remove();
        } else {
            $('tr:last', this).remove();
        }
    });
}

Netdania.Util.stripellipsis =  function(headline, length) {
    if (headline.length > length) {
        return headline.substring(0, length) + '...';
    }
    return headline;
}

var THEMES = ['white', 'black'];
Netdania.Util.getTheme = function (theme) {
    //array of themes for which css classes are defined
    if (THEMES === undefined || THEMES === null) {
        THEMES = ['white', 'black'];
    }

    var existsTheme = false;
    for (var j = 0; j < THEMES.length; j++) {
        if (THEMES[j] === theme) existsTheme = true;
    }

    //list of css classes that should be defined when adding a new theme - themeName must be: default classname + themeName (lowercase)
    //hdTDClsLeft,hdTDClsRight, TDCls,TDClsRight, nameColumn,changeColumn, nameHeaderClass, gridDefaultHeader, gridDefaultContentClass, gridDefaultContentOddClass
    //if theme is not defined use the default one
    if (existsTheme == false || theme === 'white') theme = '';

    return theme;
}

Netdania.Util.queryString = function (key, returnRowURL) {
    var page = new PageQuery(window.location.search);
    if (returnRowURL)
        return page.getValue(key);
    return unescape(page.getValue(key));
}

function PageQuery(q) {
    if (q.length > 1)
        this.q = q.substring(1, q.length);
    else this.q = null;
    this.keyValuePairs = new Array();
    if (q) {

        for (var i = 0; i < this.q.split("&").length; i++) {
            this.keyValuePairs[i] = this.q.split("&")[i];
        }
    }

    this.getKeyValuePairs = function () {
        return this.keyValuePairs;
    }

    this.getValue = function (s) {
        for (var j = 0; j < this.keyValuePairs.length; j++) {
            if (this.keyValuePairs[j].split("=")[0] == s)
                return this.keyValuePairs[j].split("=")[1];
        }
        return false;
    }

    this.getParameters = function () {

        var a = new Array(this.getLength());

        for (var j = 0; j < this.keyValuePairs.length; j++) {
            a[j] = this.keyValuePairs[j].split("=")[0];
        }
        return a;
    }

    this.getLength = function () {
        return this.keyValuePairs.length;
    }
}

function replaceQueryString(url, param, value) {
    var re = new RegExp("([?|&])" + param + "=.*?(&|$)", "i");
    if (url.match(re))
        return url.replace(re, '$1' + param + "=" + value + '$2').replace("#","");
    else
        return url.replace("#", "") + '&' + param + "=" + value;
}

function removeQueryStringParameter(url, param) {
    var re = new RegExp("([?|&])" + param + "=.*?(&|$)");
    if (url.match(re))
        return url.replace(re, '').replace("#", "");
    else
        return url;
}


function changeBackground() {
    var theme = (queryString("theme") !== 'false') ? queryString("theme").toLowerCase() : '';
    switch (theme) {
        case 'black':
            document.body.style.backgroundColor = "#000";
            var head = document.getElementsByTagName('head')[0],
                style = document.createElement('style'),
                rules = document.createTextNode('html {scrollbar-base-color:#369;scrollbar-3dlight-color:#666; scrollbar-arrow-color:#999; scrollbar-base-color:#000; scrollbar-darkshadow-color:#151515; scrollbar-face-color:#2f2f2f; scrollbar-highlight-color:#333; scrollbar-shadow-color:#151515;}');
            style.type = 'text/css';
            if (style.styleSheet)
                style.styleSheet.cssText = rules.nodeValue;
            else style.appendChild(rules);
            head.appendChild(style);

            // document.getElementsByTagName('body')[0].setAttribute("class", "black");
            // document.getElementById('wm').setAttribute("class", "white");

            break;
        default:
            //document.bgColor = "#fff";
            document.body.style.backgroundColor = "#e9e9f5";
            var head = document.getElementsByTagName('head')[0],
                style = document.createElement('style'),
                rules = document.createTextNode('html {scrollbar-face-color:#e8e8f0;' +
                    'scrollbar-track-color:#f5f5f8;' +
                    'scrollbar-arrow-color:#484848;' +
                    'scrollbar-shadow-color:#e8e8f0;' +
                    'scrollbar-dark-shadow-color:#f5f5f8;' +
                    'scrollbar-3dlight-color:#f5f5f8;' +
                    'scrollbar-highlight-color:#e8e8f0; background-color: #e9e9f5}');

            style.type = 'text/css';
            if (style.styleSheet)
                style.styleSheet.cssText = rules.nodeValue;
            else style.appendChild(rules);
            head.appendChild(style);
            break;
    }
}

function queryString(key) {
    var page = new PageQuery(window.location.search);
    return unescape(page.getValue(key));
}

/*
 decode all but the Curly Baces
 */
Netdania.Util.URLDecode = function (encodedString) {
    var output = encodedString;
    var binVal, thisString;
    var myregexp = /(%[^%|%7B|%7D]{2})/;
    while ((match = myregexp.exec(output)) != null
        && match.length > 1
        && match[1] != '') {
        binVal = parseInt(match[1].substr(1), 16);
        thisString = String.fromCharCode(binVal);
        output = output.replace(match[1], thisString);
    }
    return output;
}


Netdania.Util.stripHeadline = function (headline, length) {

    if (headline.length < length) {
        return headline;
    }
    else {
        return headline.substring(0, length) + '...';
    }
}

Netdania.Util.resetColor = function (spanid) {
    var span = document.getElementById(spanid);
    if (span) {
        for (var i = 0; i < span.children.length; i++) {
            span.children[i].style.color = '#6a6a6a';
        }
    }
}


Netdania.Util.resetHighlight = function (spanid) {
    var span = document.getElementById(spanid);
    if (span !== null) {

        for (var i = 0; i < span.children.length; i++) {
            span.children[i].style.backgroundColor = 'transparent';
        }
        Netdania.Util.resetColor(spanid);
//        setTimeout(function () {
//            Netdania.Util.resetColor(spanid);
//        }, 500);
    } else {

    }
}

Netdania.Util.overlayMask = function (elem, action) {


    var mask = elem.find('.nd-overlay-mask');

    // Create the required mask

    if (!mask.length) {
        elem.css({
            position:'relative'
        });
        mask = $('<div class="nd-overlay-mask" style="text-align:center; padding-top:0px;font-family:Arial"></div>');

        mask.css({
            position:'absolute',
            width:'100%',
            height:'100%',
            top:'0px',
            left:'0px',
            zIndex:100
        }).appendTo(elem);
    }
    $('.nd-overlay-mask').css('background', '#fff url(/JSComponents/cssgeneral/images/loading.gif) no-repeat center center');
    $('.nd-overlay-mask').css({ opacity:0.7 });
    // Act based on params

    if (!action || action === 'show') {
        mask.show();
    } else if (action === 'hide') {
        mask.hide();
    }

    return elem;
}

function formatNumber(number, nDecimalDigits) {
    //default values
    if (number !== '') {
        decimalSeparator = '.';
        thousandsSeparator = '';
        nDecimalDigits = (nDecimalDigits == undefined) ? 2 : nDecimalDigits;
        var fixed = number.toFixed(nDecimalDigits), //limit/add decimal digits
            parts = RegExp('^(-?\\d{1,3})((\\d{3})+)\\.(\\d{' + nDecimalDigits + '})$').exec(fixed); //separate begin [$1], middle [$2] and decimal digits [$4]

        if (parts) { //number >= 1000 || number <= -1000
            return parts[1] + parts[2].replace(/\d{3}/g, thousandsSeparator + '$&') + decimalSeparator + parts[4];
        } else {
            return fixed.replace('.', decimalSeparator);
        }
    }
    return '';
}


if (typeof jQuery != 'undefined') {
    $.fn.togglepanels = function () {
        return this.each(function () {
            $(this).addClass("ui-accordion ui-accordion-icons ui-widget ui-helper-reset")
                .find("h3")
                .addClass("ui-accordion-header ui-helper-reset ui-state-default ui-corner-top ui-corner-bottom")
                .hover(function () {
                    $(this).toggleClass("ui-state-hover");
                })
                .prepend('<span class="ui-icon ui-icon-triangle-1-e"></span>')
                .click(function () {
                    $(this)
                        //.toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
                        .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end()
                        .next().slideToggle();
                    return false;
                })
                .next()
            //.addClass("ui-accordion-content ui-helper-reset ui-widget-content ui-corner-bottom")
            //.hide();
        });
    };
}


Netdania.Util.Cookies = function () {
    this.getCookie = function (c_name) {
        var i, x, y, ARRcookies = document.cookie.split(";");
        for (i = 0; i < ARRcookies.length; i++) {
            x = ARRcookies[i].substr(0, ARRcookies[i].indexOf("="));
            y = ARRcookies[i].substr(ARRcookies[i].indexOf("=") + 1);
            x = x.replace(/^\s+|\s+$/g, "");
            if (x == c_name) {
                return unescape(y);
            }
        }
        return null;
    }

    this.setCookie = function (c_name, value, exdays) {
        var exdate = new Date();
        exdate.setDate(exdate.getDate() + exdays);
        var c_value = escape(value) + ((exdays == null) ? ";path=/;" : ";path=/; expires=" + exdate.toUTCString());
        document.cookie = c_name + "=" + c_value;
    }

    return this;
}

Netdania.Util.validateEmail = function (emailAddress) {
    var re = /^(([^<>()[\]\\.,;:\s@\"]+(\.[^<>()[\]\\.,;:\s@\"]+)*)|(\".+\"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    if (!re.test(emailAddress)) {
        return false;
    }
    return true;
}

Netdania.Util.PopUp = function () {
    var _htmlContent = '';
    var containerName;
    var cssClass;
    var dialogHeight = 200;
    var _self = this;
    var _buttonOk = "Ok";
    var _buttonCancel = "Cancel";
    $("#nd-customPopup-dialog-form").remove();
    this.initPopup = function (config) {
        var htmlContent = '<div id="nd-customPopup-dialog-form" class="nd-customPopup-dialog-form" title="{Title}"> ' +
            '<form>' +
            '<fieldset>' +
            '<table>' +
            '<tr>' +
            '<td>' +
            '<img class="nd-customPopup-dialog-form-image" {ImageType} />' +
            '</td>' +
            '<td>' +
            '   <div>' +
            '       <span>{Message}</span>' +
            '   </div>' +
            '</td>' +
            '</tr>' +
            '<tr>' +
            '<td colspan="2">' +
            '   <div>' +
            '        {Content}' +
            '   </div>' +
            '</td>' +
            '</tr>' +
            '</table>' +
            '</fieldset>' +
            '</form>' +
            '</div>';
        _htmlContent = htmlContent;
        if (config && config.containerName) {
            containerName = config.containerName;
            $("#" + containerName).html('');
        }
        if (config && config.cssClass) {
            cssClass = config.cssClass;
        }
    }

    this.showPopup = function (config) {

        if (config.title) {
            _htmlContent = _htmlContent.replace('{Title}', config.title);
        }
        else {
            _htmlContent = _htmlContent.replace('{Title}', '');
        }
        if (config.message) {
            _htmlContent = _htmlContent.replace('{Message}', config.message);
        }
        else {
            _htmlContent = _htmlContent.replace('{Message}', '');
        }
        if (config.content) {
            _htmlContent = _htmlContent.replace('{Content}', config.content);
        }
        else {
            _htmlContent = _htmlContent.replace('{Content}', '');
        }
        if (config.messageType) {
            _htmlContent = _htmlContent.replace('{ImageType}', "src='/JSComponents/images/" + config.messageType + ".png'");
        }
        else {
            _htmlContent = _htmlContent.replace('{ImageType}', '');
        }
        if (config.buttonsText) {
            _buttonOk = config.buttonsText[0];
            _buttonCancel = config.buttonsText[1];
        }
        var callbackFunction = function () {
            _self.closePopup();
        };
        var cancelCallbackFunction = function () {
            _self.closePopup();
        };
        if (config.callbackFunction) {
            callbackFunction = config.callbackFunction;
        }
        if (config.cancelCallbackFunction) {
            cancelCallbackFunction = config.cancelCallbackFunction;
        }

        if (config && config.dialogHeight) {
            dialogHeight = config.dialogHeight;
        }
        var buttons = [
                    {
                        text: (config.okButtonText !== undefined) ? config.okButtonText : _buttonOk,
                        click: function () {
                            callbackFunction();
                        }
                    }];
       

        if (config.viewAllButton !== undefined) {
            buttons.push({
                text: config.viewAllButton.text,
                click: function () {
                    config.viewAllButton.callbackFunction();
                    return false;
                }
            });
        }


        if (config.noCancelButton == undefined || config.noCancelButton == false) {
            buttons.push({
                text: (config.cancelButtonText !== undefined) ? config.cancelButtonText : _buttonCancel,
                click: function () {
                    cancelCallbackFunction();
                    return;
                }
            });
        }

        $("#" + containerName).html(_htmlContent);
        $("#nd-customPopup-dialog-form").dialog({
            autoOpen: false,
            //height: dialogHeight,
            width: 280,
            modal: true,
            buttons: buttons,
            dialogClass: cssClass
        }
        )
        ;
        $("#nd-customPopup-dialog-form").dialog("open");

    }
    this.closePopup = function () {
        $("#nd-customPopup-dialog-form").dialog("close");
    }
}

Netdania.Util.User = function () {
    this.loginText = 'Sign in for realtime news';

    this.isAuthenticated = function () {
        var _isAuthenticated = false;
        var _cookie = new Netdania.Util.Cookies();
        var _loginValue = _cookie.getCookie('login');

        if (_loginValue !== null && _loginValue !== '' && _loginValue !== undefined) {
            _isAuthenticated = true;
        }
        return _isAuthenticated;
    }

    this.getLoginLink = function (customLoginText) {
        var loginText = customLoginText != undefined ? customLoginText : this.loginText;
        return '<a href="/UI/UserLogin.aspx?ReturnUrl=' + encodeURIComponent(window.location.href) + '"><div class="nd-news-ticker-login"><img border="0" src="/JSComponents/CssGeneral/images/login16X16.png">' + loginText + '</div></a>';
    }

    this.getLoginURL = function () {
        return '/UI/UserLogin.aspx?ReturnUrl=' + encodeURIComponent(window.location.href);
    }

    this.getUserId = function () {
        if (this.isAuthenticated()) {
            var _cookie = new Netdania.Util.Cookies();
            var _loginValue = _cookie.getCookie('login');
            return _loginValue.split('|')[0].toLowerCase();
        } else {
            return null;
        }
    }

    this.countryCode = function () {
        var code = '';
        var codeElement = $('input[name$=userCountryCode]');
        if (codeElement != undefined && codeElement.length > 0) {
            code = codeElement.val();
            if (code == '--'){
                code = '';
            }
        }
        return code.toLowerCase();
    }

    this.countryName = function () {
        var name = '';
        var codeElement = $('input[name$=userCountryName]');
        if (codeElement != undefined && codeElement.length > 0) {
            name = codeElement.val();
            if (name == '--'){
                name = '';
            }
        }
        return name;
    }
};

Netdania.Util.Utf8_decode = function (str_data) {
    var tmp_arr = [], i = 0, ac = 0, c1 = 0, c2 = 0, c3 = 0;

    str_data += '';

    while (i < str_data.length) {
        c1 = str_data.charCodeAt(i);
        if (c1 < 128) {
            tmp_arr[ac++] = String.fromCharCode(c1);
            i++;
        } else if ((c1 > 191) && (c1 < 224)) {
            c2 = str_data.charCodeAt(i + 1);
            tmp_arr[ac++] = String.fromCharCode(((c1 & 31) << 6) | (c2 & 63));
            i += 2;
        } else {
            c2 = str_data.charCodeAt(i + 1);
            c3 = str_data.charCodeAt(i + 2);
            tmp_arr[ac++] = String.fromCharCode(((c1 & 15) << 12) | ((c2 & 63) << 6) | (c3 & 63));
            i += 3;
        }
    }

    return tmp_arr.join('');
}

Netdania.Util.InstrumentTypes = { Forex:'FOREX', Stocks:'STOCKS' };

Netdania.Util.GetInstrumentType = function (symbol, isin) {
    if (isin !== '' || isin !== undefined) {
        if (isin.indexOf('FX') > -1)
            return Netdania.Util.InstrumentTypes.Forex;
        else
            return Netdania.Util.InstrumentTypes.Stocks;
    }
    return Netdania.Util.InstrumentTypes.Forex;
}

Netdania.Util.LoadScriptOnDemand = function (url) {
    $.ajax({
        url:url,
        dataType:"script",
        async:false
    });
}
//next we load the translation class for english
Netdania.Util.GetTranslation = function (key) {
    var translations = new Netdania.TranslationsEN();
    return translations.MESSAGES[key];

}


Netdania.Util.getIntrumentRealType = function (data) {

    var hasLast = false;
    var hasBid = false;
    var hasAsk = false;

    if (data == null)
        return -1;

    if (!(data.get(Netdania.JsApi.Fields.QUOTE_LAST) == "N/A"))
        hasLast = true;

    if (!(data.get(Netdania.JsApi.Fields.QUOTE_BID) == "N/A"))
        hasBid = true;

    if (!(data.get(Netdania.JsApi.Fields.QUOTE_ASK) == "N/A"))
        hasAsk = true;

    if (hasLast && hasBid && hasAsk)
        return 2;

    if (hasBid && hasAsk)
        return 0;

    if (hasLast)
        return 1;

    return -1;
}


var linksArrayStocks = [

     { link: '/Products/QuoteList/Stocks/Austria.aspx?tm=europe-austria', name: 'Austria' },
        { link: '/Products/QuoteList/Stocks/Belgium.aspx?tm=europe-belgium', name: 'Belgium' },
        { link: '/Products/QuoteList/Stocks/Denmark.aspx?tm=europe-denmark', name: 'Denmark' },
        { link: '/Products/QuoteList/Stocks/Finland.aspx?tm=europe-finland', name: 'Finland' },
        { link: '/Products/QuoteList/Stocks/France.aspx?tm=europe-france', name: 'France' },
        { link: '/Products/QuoteList/Stocks/Germany.aspx?tm=europe-germany', name: 'Germany' },
        { link: '/Products/QuoteList/Stocks/India.aspx?tm=india', name: 'India' },
        { link: '/Products/QuoteList/Stocks/Ireland.aspx?tm=europe-ireland', name: 'Ireland' },
        { link: '/Products/QuoteList/Stocks/Italy.aspx?tm=europe-italy', name: 'Italy' },
        { link: '/Products/QuoteList/Stocks/Netherlands.aspx?tm=europe-netherlands', name: 'Netherlands' },
        { link: '/Products/QuoteList/Stocks/Norway.aspx?tm=europe-norway', name: 'Norway' },
        { link: '/Products/QuoteList/Stocks/Portugal.aspx?tm=europe-portugal', name: 'Portugal' },
        { link: '/Products/QuoteList/Stocks/Spain.aspx?tm=europe-spain', name: 'Spain' },
        { link: '/Products/QuoteList/Stocks/Sweden.aspx?tm=europe-sweden', name: 'Sweden' },
        { link: '/Products/QuoteList/Stocks/Switzerland.aspx?tm=europe-switzerland', name: 'Switzerland' },
        { link: '/Products/QuoteList/Stocks/Turkey.aspx?tm=europe-turkey', name: 'Turkey' },
        { link: '/Products/QuoteList/Stocks/UK.aspx?tm=europe-uk', name: 'UK' },
        { link: '/Products/QuoteList/Stocks/USA.aspx?tm=usa', name: 'USA' }
];


var linksArrayStocksEurope = [

        { link: '/Products/QuoteList/Stocks/Austria.aspx?tm=europe-austria', name: 'Austria' },
        { link: '/Products/QuoteList/Stocks/Belgium.aspx?tm=europe-belgium', name: 'Belgium' },
        { link: '/Products/QuoteList/Stocks/Denmark.aspx?tm=europe-denmark', name: 'Denmark' },
        { link: '/Products/QuoteList/Stocks/Finland.aspx?tm=europe-finland', name: 'Finland' },
        { link: '/Products/QuoteList/Stocks/France.aspx?tm=europe-france', name: 'France' },
        { link: '/Products/QuoteList/Stocks/Germany.aspx?tm=europe-germany', name: 'Germany' },
        { link: '/Products/QuoteList/Stocks/India.aspx?tm=india', name: 'India' },
        { link: '/Products/QuoteList/Stocks/Ireland.aspx?tm=europe-ireland', name: 'Ireland' },
        { link: '/Products/QuoteList/Stocks/Italy.aspx?tm=europe-italy', name: 'Italy' },
         
        { link: '/Products/QuoteList/Stocks/Netherlands.aspx?tm=europe-netherlands', name: 'Netherlands' },
        { link: '/Products/QuoteList/Stocks/Norway.aspx?tm=europe-norway', name: 'Norway' },
        { link: '/Products/QuoteList/Stocks/Portugal.aspx?tm=europe-portugal', name: 'Portugal' },
        { link: '/Products/QuoteList/Stocks/Spain.aspx?tm=europe-spain', name: 'Spain' },
        { link: '/Products/QuoteList/Stocks/Sweden.aspx?tm=europe-sweden', name: 'Sweden' },
        { link: '/Products/QuoteList/Stocks/Switzerland.aspx?tm=europe-switzerland', name: 'Switzerland' },
        { link: '/Products/QuoteList/Stocks/Turkey.aspx?tm=europe-turkey', name: 'Turkey' },
        { link: '/Products/QuoteList/Stocks/UK.aspx?tm=europe-uk', name: 'UK' },
       
        { link: '/Products/QuoteList/Stocks/USA.aspx?tm=usa', name: 'USA' }
];


var linksArrayMajors = [
                            { link: '/Products/QuoteList/LocalCurrencies.aspx?m=q&tm=local', name: 'DKK' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=majors', name: 'Majors' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=dollar', name: 'Dollar' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=euro', name: 'Euro' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=sterling', name: 'Sterling' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=usdforwards', name: 'USD Forwards' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&lite=no&tm=ql', name: 'QuoteList' }
                            ];

var linksArrayDollar = [{ link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=dollar-majors', name: 'Majors' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=dollar-asia', name: 'Asia Pacific' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=dollar-south_america', name: 'South America' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=dollar-central_america', name: 'Central America' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=dollar-western_europe', name: 'Western Europe' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=dollar-eastern_europe', name: 'Eastern Europe' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=dollar-middle_east', name: 'Middle East' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=dollar-africa', name: 'Africa' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=dollar-others', name: 'Others' }
                            ];

var linksArrayEuro = [{ link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=euro-majors', name: 'Majors' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=euro-asia', name: 'Asia Pacific' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=euro-south_america', name: 'South America' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=euro-central_america', name: 'Central America' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=euro-western_europe', name: 'Western Europe' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=euro-eastern_europe', name: 'Eastern Europe' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=euro-middle_east', name: 'Middle East' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=euro-africa', name: 'Africa' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=euro-others', name: 'Others' }
                            ];

var linksArraySterling = [{ link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=sterling-majors', name: 'Majors' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=sterling-asia', name: 'Asia Pacific' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=sterling-south_america', name: 'South America' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=sterling-central_america', name: 'Central America' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=sterling-western_europe', name: 'Western Europe' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=sterling-eastern_europe', name: 'Eastern Europe' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=sterling-middle_east', name: 'Middle East' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=sterling-africa', name: 'Africa' },
                            { link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=sterling-others', name: 'Others' }
                            ];

var linksArrayForwards = [{ link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=usdforwards', name: 'USD' }

                            ];

var linksArrayMetals = [{ link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=preciousmetals', name: 'Precious' }

                            ];

var linksArrayWorldMarkets = [{ link: '/Products/live-streaming-currency-rates-foreign-exchange/real-time-quotes/QuoteList.aspx?m=q&tm=worldmarkets', name: 'Global' }

                            ];

function createLinks(linksArray, selectedPosition, tabName, parentLink, parentLinkName, menuName, itemsOnLine) {

    $('.nd-news-menu').append('<div id="nd-sup-menu" style="padding-bottom:7px;padding-left:7px"></div>');
    if (itemsOnLine) {
        $('.nd-news-menu').append('<div id="nd-sup-menu1" style="padding-bottom:7px;padding-left:7px"></div>');
        for (var i = 0; i < itemsOnLine; i++) {
            if (i == selectedPosition) {
                $('#nd-sup-menu').append('<a class="nd-news-menu-selected" href="' + linksArray[i].link + '"> ' + linksArray[i].name + '</a>   &nbsp;');
            } else {
                $('#nd-sup-menu').append('<a class="nd-news-menu-unselected" href="' + linksArray[i].link + '"> ' + linksArray[i].name + '</a>  &nbsp;');
            }
        }
        for (var i = itemsOnLine; i < linksArray.length; i++) {
            if (i == selectedPosition) {
                $('#nd-sup-menu1').append('<a class="nd-news-menu-selected" href="' + linksArray[i].link + '"> ' + linksArray[i].name + '</a>   &nbsp;');
            } else {
                $('#nd-sup-menu1').append('<a class="nd-news-menu-unselected" href="' + linksArray[i].link + '"> ' + linksArray[i].name + '</a>  &nbsp;');
            }
        }
    }
    else {
        for (var i = 0; i < linksArray.length; i++) {
            if (i == selectedPosition) {
                $('#nd-sup-menu').append('<a class="nd-news-menu-selected" href="' + linksArray[i].link + '"> ' + linksArray[i].name + '</a>   &nbsp;');
            } else {
                $('#nd-sup-menu').append('<a class="nd-news-menu-unselected" href="' + linksArray[i].link + '"> ' + linksArray[i].name + '</a>  &nbsp;');
            }
        }

    }


    if (parentLink) {
        $('.nd-news-menu').append('<div id="nd-sub-menu" style="border-bottom:0px solid #666; padding-bottom:4px; background: #eee; padding-top:3px;"></div>');

        $('#nd-sub-menu').append('<span class="nd-current-menu" style="font-weight:normal !important;"><a href="' + parentLink + '">' + parentLinkName + '</a></span>');
    }

    if (tabName !== '') {
        $('#nd-sub-menu').append('<span style="margin-left:4px; margin-right:2px; font-size: 11px !important">»</span><span class="nd-current-menu" style="font-weight:normal !important;">' + tabName + '</span>');
    }

    if (menuName) {
        $('#nd-sub-menu').append('<span style="font-size:11px;  margin-left:2px; margin-right:0px">»</span><span class="nd-current-menu">' + menuName + '</span>');
    }
}


function getIndexOfCountryStock(key) {
    for (var i = 0; i < linksArrayStocksEurope.length; i++) {
        if (linksArrayStocksEurope[i].name == key)
            return i;
    }
    return 0;
}

function createMenu(tab) {

    var forexMenuLink = "/Products/QuoteList/LocalCurrencies.aspx?m=q&tm=local";
    var forexMenuName = "Forex";

    var stocksMenuLink = "/Products/QuoteList/Stocks/CountrySpecific.aspx?tm=country";
    var stocksMenuName = "Stocks";

    switch (tab) {
        case 'local':
            createLinks(linksArrayMajors, 0, '');
            break;
        case 'majors':
            createLinks(linksArrayMajors, 1, '');
            break;
        case 'dollar':
            createLinks(linksArrayDollar, 0, 'Dollar', forexMenuLink, forexMenuName, 'Majors');
            break;
        //dollar    
        case 'dollar-majors':
            createLinks(linksArrayDollar, 0, 'Dollar', forexMenuLink, forexMenuName, 'Majors');
            break;
        case 'dollar-asia':
            createLinks(linksArrayDollar, 1, 'Dollar', forexMenuLink, forexMenuName, 'Asia');
            break;
        case 'dollar-south_america':
            createLinks(linksArrayDollar, 2, 'Dollar', forexMenuLink, forexMenuName, 'South America');
            break;
        case 'dollar-central_america':
            createLinks(linksArrayDollar, 3, 'Dollar', forexMenuLink, forexMenuName, 'Central America');
            break;
        case 'dollar-western_europe':
            createLinks(linksArrayDollar, 4, 'Dollar', forexMenuLink, forexMenuName, 'Western Europe');
            break;
        case 'dollar-eastern_europe':
            createLinks(linksArrayDollar, 5, 'Dollar', forexMenuLink, forexMenuName, 'Eastern Europe');
            break;
        case 'dollar-middle_east':
            createLinks(linksArrayDollar, 6, 'Dollar', forexMenuLink, forexMenuName, 'Middle East');
            break;
        case 'dollar-africa':
            createLinks(linksArrayDollar, 7, 'Dollar', forexMenuLink, forexMenuName, 'Africa');
            break;
        case 'dollar-others':
            createLinks(linksArrayDollar, 8, 'Dollar', forexMenuLink, forexMenuName, 'Dollar');
            break;
        //euro         
        case 'euro':
            createLinks(linksArrayEuro, 0, 'Euro', forexMenuLink, forexMenuName, 'Majors');
            break;
        case 'euro-majors':
            createLinks(linksArrayEuro, 0, 'Euro', forexMenuLink, forexMenuName, 'Majors');
            break;
        case 'euro-asia':
            createLinks(linksArrayEuro, 1, 'Euro', forexMenuLink, forexMenuName, 'Asia');
            break;
        case 'euro-south_america':
            createLinks(linksArrayEuro, 2, 'Euro', forexMenuLink, forexMenuName, 'South America');
            break;
        case 'euro-central_america':
            createLinks(linksArrayEuro, 3, 'Euro', forexMenuLink, forexMenuName, 'Central America');
            break;
        case 'euro-western_europe':
            createLinks(linksArrayEuro, 4, 'Euro', forexMenuLink, forexMenuName, 'Western Europe');
            break;
        case 'euro-eastern_europe':
            createLinks(linksArrayEuro, 5, 'Euro', forexMenuLink, forexMenuName, 'Eastern Europe');
            break;
        case 'euro-middle_east':
            createLinks(linksArrayEuro, 6, 'Euro', forexMenuLink, forexMenuName, 'Middle East');
            break;
        case 'euro-africa':
            createLinks(linksArrayEuro, 7, 'Euro', forexMenuLink, forexMenuName, 'Africa');
            break;
        case 'euro-others':
            createLinks(linksArrayEuro, 8, 'Euro', forexMenuLink, forexMenuName, 'Others');
            break;
        //sterling          
        case 'sterling':
            createLinks(linksArraySterling, 0, 'Sterling', forexMenuLink, forexMenuName, 'Majors');
            break;
        case 'sterling-majors':
            createLinks(linksArraySterling, 0, 'Sterling', forexMenuLink, forexMenuName, 'Majors');
            break;
        case 'sterling-asia':
            createLinks(linksArraySterling, 1, 'Sterling', forexMenuLink, forexMenuName, 'Asia');
            break;
        case 'sterling-south_america':
            createLinks(linksArraySterling, 2, 'Sterling', forexMenuLink, forexMenuName, 'South America');
            break;
        case 'sterling-central_america':
            createLinks(linksArraySterling, 3, 'Sterling', forexMenuLink, forexMenuName, 'Central America');
            break;
        case 'sterling-western_europe':
            createLinks(linksArraySterling, 4, 'Sterling', forexMenuLink, forexMenuName, 'Western Europe');
            break;
        case 'sterling-eastern_europe':
            createLinks(linksArraySterling, 5, 'Sterling', forexMenuLink, forexMenuName, 'Eastern Europe');
            break;
        case 'sterling-middle_east':
            createLinks(linksArraySterling, 6, 'Sterling', forexMenuLink, forexMenuName, 'Middle East');
            break;
        case 'sterling-africa':
            createLinks(linksArraySterling, 7, 'Sterling', forexMenuLink, forexMenuName, 'Africa');
            break;
        case 'sterling-others':
            createLinks(linksArraySterling, 8, 'Sterling', forexMenuLink, forexMenuName, 'Others');
            break;

        //     
        case 'usdforwards':
            createLinks(linksArrayMajors, 5, '');
            break;
        case 'ql':
            createLinks(linksArrayMajors, 6, '');
            break;
        //preciousmetals     
        case 'preciousmetals':
            createLinks(linksArrayMetals, 0, 'Metals');
            break;

        //worldmarkets      
        case 'worldmarkets':
            createLinks(linksArrayWorldMarkets, 0, 'Indices');
            break;

        case 'country':
            createLinks(linksArrayStocks, 0, '');
            break;
        case 'Stocks':
            createLinks(linksArrayStocks, 1, '');
            break;
        case 'usa':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('USA'), '', undefined, undefined, undefined, linksArrayStocksEurope.length);
            break;
        case 'europe-austria':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('Austria'), '', undefined, undefined, undefined, linksArrayStocksEurope.length); //'Europe', stocksMenuLink, stocksMenuName, 'Austria');
            break;
        case 'europe-belgium':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('Belgium'), '', undefined, undefined, undefined, linksArrayStocksEurope.length); //'Europe', stocksMenuLink, stocksMenuName, 'Belgium');
            break;
        case 'europe-denmark':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('Denmark'), '', undefined, undefined, undefined, linksArrayStocksEurope.length); //'Europe', stocksMenuLink, stocksMenuName, 'Denmark');
            break;
        case 'europe-finland':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('Finland'), '', undefined, undefined, undefined, linksArrayStocksEurope.length);  //'Europe', stocksMenuLink, stocksMenuName, 'Finland');
            break;
        case 'europe-france':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('France'), '', undefined, undefined, undefined, linksArrayStocksEurope.length);  //'Europe', stocksMenuLink, stocksMenuName, 'France');
            break;
        case 'europe-germany':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('Germany'), '', undefined, undefined, undefined, linksArrayStocksEurope.length); //'Europe', stocksMenuLink, stocksMenuName, 'Germany');
            break;
        case 'europe-ireland':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('Ireland'), '', undefined, undefined, undefined, linksArrayStocksEurope.length);  //'Europe', stocksMenuLink, stocksMenuName, 'Ireland');
            break;
        case 'europe-italy':
            createLinks(linksArrayStocksEurope,getIndexOfCountryStock('Italy'), '', undefined, undefined, undefined, linksArrayStocksEurope.length);  //'Europe', stocksMenuLink, stocksMenuName, 'Italy');
            break;
        case 'europe-netherlands':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('Netherlands'), '', undefined, undefined, undefined, linksArrayStocksEurope.length);  //'Europe', stocksMenuLink, stocksMenuName, 'Netherlands');
            break;
        case 'europe-norway':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('Norway'), '', undefined, undefined, undefined, linksArrayStocksEurope.length);  //'Europe', stocksMenuLink, stocksMenuName, 'Norway');
            break;
        case 'europe-portugal':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('Portugal'), '', undefined, undefined, undefined, linksArrayStocksEurope.length);  //'Europe', stocksMenuLink, stocksMenuName, 'Portugal');
            break;
        case 'europe-spain':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('Spain'), '', undefined, undefined, undefined, linksArrayStocksEurope.length);  //'Europe', stocksMenuLink, stocksMenuName, 'Spain');
            break;
        case 'europe-sweden':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('Sweden'), '', undefined, undefined, undefined, linksArrayStocksEurope.length);  //'Europe', stocksMenuLink, stocksMenuName, 'Sweden');
            break;
        case 'europe-switzerland':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('Switzerland'), '', undefined, undefined, undefined, linksArrayStocksEurope.length);  //'Europe', stocksMenuLink, stocksMenuName, 'Switzerland');
            break;
        case 'europe-uk':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('UK'), '', undefined, undefined, undefined, linksArrayStocksEurope.length);  //'Europe', stocksMenuLink, stocksMenuName, 'UK');
            break;
        case 'india':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('India'), '', undefined, undefined, undefined, linksArrayStocksEurope.length);
            break;
        case 'europe-turkey':
            createLinks(linksArrayStocksEurope, getIndexOfCountryStock('Turkey'), '', undefined, undefined, undefined, linksArrayStocksEurope.length);
            break;
    }
}


Netdania.Util.getCountryISOCode = function() {
    return 'DK';
}

Netdania.Util.isNumber = function(n) {
    return !isNaN(parseFloat(n)) && isFinite(n);
}

Netdania.Util.getLocation = function(href) {
    var location = document.createElement("a");
    location.href = href;
    // IE doesn't populate all link properties when setting .href with a relative URL,
    // however .href will return an absolute URL which then can be used on itself
    // to populate these additional fields.
    if (location.host == "") {
        location.href = location.href;
    }
    return location;
};


Netdania.Util.Utf8 = {
    // public method for url encoding
    encode : function (string) {
        string = string.replace(/\r\n/g,"\n");
        var utftext = "";
        for (var n = 0; n < string.length; n++) {
            var c = string.charCodeAt(n);
            if (c < 128) {
                utftext += String.fromCharCode(c);
            }
            else if((c > 127) && (c < 2048)) {
                utftext += String.fromCharCode((c >> 6) | 192);
                utftext += String.fromCharCode((c & 63) | 128);
            }
            else {
                utftext += String.fromCharCode((c >> 12) | 224);
                utftext += String.fromCharCode(((c >> 6) & 63) | 128);
                utftext += String.fromCharCode((c & 63) | 128);
            }
        }
        return utftext;
    },
    // public method for url decoding
    decode : function (utftext) {
        var string = "";
        var i = 0;
        var c = c1 = c2 = 0;
        while ( i < utftext.length ) {
            c = utftext.charCodeAt(i);
            if (c < 128) {
                string += String.fromCharCode(c);
                i++;
            }
            else if((c > 191) && (c < 224)) {
                c2 = utftext.charCodeAt(i+1);
                string += String.fromCharCode(((c & 31) << 6) | (c2 & 63));
                i += 2;
            }
            else {
                c2 = utftext.charCodeAt(i+1);
                c3 = utftext.charCodeAt(i+2);
                string += String.fromCharCode(((c & 15) << 12) | ((c2 & 63) << 6) | (c3 & 63));
                i += 3;
            }
        }
        return string;
    }
};

Netdania.Util.Round = function(value, exp) {
    if (typeof exp === 'undefined' || +exp === 0)
        return Math.round(value);

    value = +value;
    exp  = +exp;

    if (isNaN(value) || !(typeof exp === 'number' && exp % 1 === 0))
        return NaN;

    // Shift
    value = value.toString().split('e');
    value = Math.round(+(value[0] + 'e' + (value[1] ? (+value[1] + exp) : exp)));

    // Shift back
    value = value.toString().split('e');
    return +(value[0] + 'e' + (value[1] ? (+value[1] - exp) : -exp));
};

Netdania.Util.HashCode = function(s){
    var hash = 0;
    var i;
    var charH;
    if (s.length == 0) return hash;
    for (i = 0, l = s.length; i < l; i++) {
        charH  = s.charCodeAt(i);
        hash  = ((hash<<5)-hash)+charH;
        hash |= 0; // Convert to 32bit integer
    }
    return Math.abs(hash);
};

Netdania.Util.addParameter = function(url, parameterName, parameterValue, atStart){
    replaceDuplicates = true;
    if(url.indexOf('#') > 0){
        var cl = url.indexOf('#');
        urlhash = url.substring(url.indexOf('#'),url.length);
    } else {
        urlhash = '';
        cl = url.length;
    }
    sourceUrl = url.substring(0,cl);

    var urlParts = sourceUrl.split("?");
    var newQueryString = "";

    if (urlParts.length > 1)
    {
        var parameters = urlParts[1].split("&");
        for (var i=0; (i < parameters.length); i++)
        {
            var parameterParts = parameters[i].split("=");
            if (!(replaceDuplicates && parameterParts[0] == parameterName))
            {
                if (newQueryString == "")
                    newQueryString = "?";
                else
                    newQueryString += "&";
                newQueryString += parameterParts[0] + "=" + (parameterParts[1]?parameterParts[1]:'');
            }
        }
    }
    if (newQueryString == "")
        newQueryString = "?";

    if(atStart){
        newQueryString = '?'+ parameterName + "=" + parameterValue + (newQueryString.length>1?'&'+newQueryString.substring(1):'');
    } else {
        if (newQueryString !== "" && newQueryString != '?')
            newQueryString += "&";
        newQueryString += parameterName + "=" + (parameterValue?parameterValue:'');
    }
    return urlParts[0] + newQueryString + urlhash;
};


Netdania.Util.scrollToElement = function(container, selector, time, verticalOffset) {
    time = typeof(time) != 'undefined' ? time : 1000;
    verticalOffset = typeof(verticalOffset) != 'undefined' ? verticalOffset : 0;
    var element = $(selector);
    var parent = $(container);
    var offset = element.offset();
    var offsetparent = parent.offset();
    var parentScroll = (parent.scrollTop() ?  parent.scrollTop() : 0);

    offsetTop = offset.top - offsetparent.top + parentScroll + verticalOffset;
    parent.animate({
        scrollTop: offsetTop
    }, time);
};

Netdania.Util.isSecure = function () {
    return document.location.protocol == 'https:';
};

Netdania.Util.CountDecimals = function (value) {
    if (Math.floor(value) !== value)
        return value.toString().split(".")[1].length || 0;
    return 0;
};

Netdania.Util.Capitalize = function (s) {
    return s[0].toUpperCase() + s.toLowerCase().slice(1);
};

Netdania.Util.LogActive = true;

Netdania.Util.LogError = function (message, error) {
    var error_message = "";
    if (error !== undefined && error != null) {
        error_message = error;
        if (error.message !== undefined && error.message != null) {
            error_message = error.message;
        }
    }
    Netdania.Util.Log(message, error, Netdania.Util.LogActive);
};

Netdania.Util.LogInfo = function (data) {
    if (true) Netdania.Util.Log(data, "", Netdania.Util.LogActive);
};

Netdania.Util.Log = function (message, error, isLogActive) {

    var active = isLogActive || false;
    if (isLogActive) {

        if (typeof console == "undefined") {
            this.console = { log: function () { } };
        }

        console.log(message + " " + error);
        var active = isLogActive || false;
        if (isLogActive) {
        }
    }
};
