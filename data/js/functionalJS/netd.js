function getReqObjStory(strStoryId, strProvider, history) {

    var s = '{"StoryId":' + strStoryId + ', "History":' + history.toString() + '}';
    return Netdania.JsApi.Request.getReqObjStory(s, strProvider);;
}

var dateFormat = function () {
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
        var dF = dateFormat;

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
            d: d,
            dd: pad(d),
            ddd: dF.i18n.dayNames[D],
            dddd: dF.i18n.dayNames[D + 7],
            m: m + 1,
            mm: pad(m + 1),
            mmm: dF.i18n.monthNames[m],
            mmmm: dF.i18n.monthNames[m + 12],
            yy: String(y).slice(2),
            yyyy: y,
            h: H % 12 || 12,
            hh: pad(H % 12 || 12),
            H: H,
            HH: pad(H),
            M: M,
            MM: pad(M),
            s: s,
            ss: pad(s),
            l: pad(L, 3),
            L: pad(L > 99 ? Math.round(L / 10) : L),
            t: H < 12 ? "a" : "p",
            tt: H < 12 ? "am" : "pm",
            T: H < 12 ? "A" : "P",
            TT: H < 12 ? "AM" : "PM",
            Z: utc ? "UTC" : (String(date).match(timezone) || [""]).pop().replace(timezoneClip, ""),
            o: (o > 0 ? "-" : "+") + pad(Math.floor(Math.abs(o) / 60) * 100 + Math.abs(o) % 60, 4),
            S: ["th", "st", "nd", "rd"][d % 10 > 3 ? 0 : (d % 100 - d % 10 != 10) * d % 10]
        };

        return mask.replace(token, function ($0) {
            return $0 in flags ? flags[$0] : $0.slice(1, $0.length - 1);
        });
    };
}();

// Some common format strings
dateFormat.masks = {
    "default": "ddd mmm dd yyyy HH:MM:ss",
    shortDate: "m/d/yy",
    mediumDate: "mmm d, yyyy",
    longDate: "mmmm d, yyyy",
    fullDate: "dddd, mmmm d, yyyy",
    shortTime: "h:MM TT",
    mediumTime: "h:MM:ss TT",
    longTime: "h:MM:ss TT Z",
    isoDate: "yyyy-mm-dd",
    isoTime: "HH:MM:ss",
    isoDateTime: "yyyy-mm-dd'T'HH:MM:ss",
    isoUtcDateTime: "UTC:yyyy-mm-dd'T'HH:MM:ss'Z'"
};

// Internationalization strings
dateFormat.i18n = {
    dayNames: [
    "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat",
    "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
    ],
    monthNames: [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"
    ]
};

// For convenience...
Date.prototype.format = function (mask, utc) {
    return dateFormat(this, mask, utc);
};


var globalCurrentCmp = globalCurrentCmp || 0;
var cbfunctions = cbfunctions || [];
var tid; // = setTimeout('refreshAfterTimer(\'netd-newsGrid\', true,\'current\', true )', 180000);
_FrontPageBox_baseUrl = "http://penge.netdania.com/PengeImage.aspx?instr={INSTR}&prv={PRV}&points={PT}&ts={TS}&current={CURR}&w={W}&h={H}";


function removeArrayItem(arr, v) {
    for (var i = 0; i < arr.length; i++) {
        if (arr[i] === v) {
            arr.splice(i, 1);
            break;
        }
    }
}


/**
* Gets the current time in milliseconds from 1970.
*/
function getCurrentDateTime() {
    var date = new Date();
    return Date.UTC(date.getFullYear(), date.getMonth(), date.getDate(), date.getHours(), date.getMinutes(), date.getSeconds());
}


function getDate(val) {
    dtArray = val.split(" ");
    dt = dtArray[0];
    time = dtArray[1];

    timeArray = time.split(":");
    hour = timeArray[0];
    min = timeArray[1];
    sec = timeArray[2];


    dtArray = dt.split("-");

    today = new Date();
    today.setFullYear(dtArray[0], dtArray[1], dtArray[2]);
    today.setHours(hour, min, sec);
    newDate = today;

    return newDate;
}

function isInArray(val, arr) {
    var isInArr = false;
    for (i = 0; i < arr.length; i++) {
        if (arr[i] == val) {
            isInArr = true;
            break;
        }
    }
    return isInArr;
}


function stripHeadline(headline, length) {

    if (headline.length < length) {
        return headline;
    } else {
        return headline.substring(0, length) + '...';
    }
}
function stripellipsis(headline, length) {
    if (ua.indexOf(" firefox/") >= 0 || ua.indexOf(' gecko/') >= 0 || ua.indexOf("opera/") >= 0) {
        if (headline.length > length) {
            return headline.substring(0, length) + '...';
        }
    }
    return headline;
}

function replaceHtml(el, html) {
    if (el === undefined || el === null) {
        return;
    }
    var oldEl = typeof el === "string" ? document.getElementById(el) : el;
    /*@cc_on // Pure innerHTML is slightly faster in IE
        oldEl.innerHTML = html;
        return oldEl;
        @*/

    var newEl = oldEl.cloneNode(false);
    newEl.innerHTML = html;
    oldEl.parentNode.replaceChild(newEl, oldEl);
    /* Since we just removed the old element from the DOM, return a reference
        to the new element, which can be used to restore variable references. */
    return newEl;
}

function renderTable(tableConfig, startFrame, endFrame) {
    var headerClass, contentClass, contentOddClass, width, fields, instr;

    fields = tableConfig.fields;
    var cmp = tableConfig.instr;
    instr = cmp.records;
    var tableWidth = 0;
    var headerHtml = '';
    var colsHtml = '';
    var lastField = '';
    contentClass = tableConfig.contentClass;
    if (tableConfig.contentClass === undefined || tableConfig.contentClass === null || tableConfig.contentClass === '') {
        contentClass = "gridDefaultContentClass-calendar";
    }
    contentOddClass = tableConfig.contentOddClass;
    if (tableConfig.contentOddClass === undefined || tableConfig.contentOddClass === null || tableConfig.contentOddClass === '') {
        contentOddClass = "gridDefaultContentOddClass-calendar";
    }

    if (tableConfig.width !== undefined && tableConfig.width !== null) {
        tableWidth = width;
    }
    for (var prop0 in fields) {
        if (fields.hasOwnProperty(prop0)) {
            lastField = prop0;
        }
    }
    var cols = 0;
    for (var prop in fields) {
        if (fields.hasOwnProperty(prop)) {
            cols++;
            colsHtml += '<col width=' + fields[prop].width + '>';
            tableWidth += parseInt(fields[prop].width, 10);
            var lc = '';
            if (lastField === prop) {
                lc = ' lastColCls';
            }
            headerHtml += '<th class="' + fields[prop].hdTDcls + lc + '"><span class="' + fields[prop].hdCellCls + '"><span>' + fields[prop].name + '</span></span></th>';
        }
    }
    var forceHeight = '';
    if (tableConfig.height !== undefined) {
        forceHeight = 'style="overflow: auto;overflow-x: hidden; height: ' + tableConfig.height + 'px;"';
    }
    var align = '';
    if (tableConfig.align === '') {
        align = "center";
    } else {
        align = tableConfig.align;
    }
    var jshtml = '';
    if (startFrame !== undefined) {
        jshtml = startFrame + '<table id="e-cal-table" width=' + tableWidth + ' style="table-layout:fixed;" border="0" cellspacing="0" cellpadding="0" align="' + align + '" class="e-cal-table"> ';
    } else {
        jshtml = '<table id="e-cal-table" width=' + tableWidth + ' style="table-layout:fixed;" border="0" cellspacing="0" cellpadding="0"" class="e-cal-table">';
    }
    jshtml += colsHtml;
    if (tableConfig.hideHeaders !== true) {
        jshtml += '<thead><tr>';
        jshtml += headerHtml;
        jshtml += '</tr></thead>';
    }
    jshtml += '<tbody >';
    var gday = - 1;
    var gmonth = - 1;
    var gyear = - 1;
    if (tableConfig.groupByDate === true && instr.length > 0) {
        gd = new Date(instr[0].data.t * 1000);
    }

    var noDataMessage = tableConfig.noDataMessage;
    if (noDataMessage === undefined) {
        //noDataMessage = 'There are no events scheduled!';
        noDataMessage = "";
    }

    if (instr.length === 0) {
        jshtml += '<tr><td colspan=' + cols + ' class="netd-grid-empty">' + noDataMessage + '</td></tr>';
    }
    for (var ii = 0; ii < instr.length; ii++) {
        //        if(tableConfig.groupByDate===true){
        //			var cd = new Date(instr[ii].data.t * 1000);
        //			if(cd.getDay()!==gday || cd.getMonth()!==gmonth || cd.getYear()!==gyear){
        //				jshtml += '<tr><td colspan='+cols+' > <div class=\'e-cal-date\'><span>'+dateFormat(cd, "dddd")+'<br />'+dateFormat(cd, "mmmm dd")+'</span></div></td></tr>';
        //				gday = cd.getDay();
        //				gmonth = cd.getMonth();
        //				gyear = cd.getYear();
        //			}
        //		}

        var classStyle = contentOddClass;
        if (ii % 2 === 0) {
            classStyle = contentClass;
        }
        if (instr[ii].data.s != 1) {
            jshtml += '<tr id="' + instr[ii].data.i + '" class="' + classStyle + '">';
        } else {
            var title = instr[ii].data.e;
            if (instr[ii].data.realDate !== undefined) {
                if (instr[ii].data.reccurenceType == 'week') {
                    var dateUnf = new Date(instr[ii].data.realDate * 1000);
                    if (dateUnf.getFullYear() == new Date().getFullYear())
                        title = title + '(' + dateFormat(dateUnf, "mmm dd").toUpperCase() + ')';
                    else
                        title = title + '(' + dateFormat(dateUnf, "mmm dd, yyyy").toUpperCase() + ')';
                } else {
                    var dateUnf = new Date(instr[ii].data.realDate * 1000);
                    if (dateUnf.getFullYear() == new Date().getFullYear())
                        title = title + '(' + dateFormat(dateUnf, "mmm").toUpperCase() + ')';
                    else
                        title = title + '(' + dateFormat(dateUnf, "mmm, yyyy").toUpperCase() + ')';
                }
            }
            //var clickStr = (tableConfig.click == false) ? '' : 'onclick="showStory(this,\'' + escape(title) + '\', \'' + instr[ii].data.i + '\',\'' + tableConfig.provider + '\',\'' + instr[ii].data.a + '\',\'' + instr[ii].data.p + '\',\'' + instr[ii].data.f + '\',\'' + instr[ii].data.c + '\',\'' + instr[ii].data.parentID + '\');"';
            var clickStr = (tableConfig.click == false) ? '' : 'onclick="showStory(this,\'' + escape(title) + '\', \'' + instr[ii].data.i + '\',\'' + tableConfig.provider + '\',\'' + instr[ii].data.a + '\',\'' + instr[ii].data.p + '\',\'' + instr[ii].data.f + '\',\'' + instr[ii].data.c + '\',\'' + instr[ii].data.parentID + '\');ChangeHistoryTab(\'Chart' + instr[ii].data.i + '\', \'' + instr[ii].data.i + '\', \'' + tableConfig.provider + '\', \'' + instr[ii].data.parentID + '\');"';
            jshtml += '<tr id="' + instr[ii].data.i + '" class="' + classStyle + '" style="border-top:1px dotted black;" ' + clickStr + ' >';
        }

        var recid = instr[ii].id;

        for (var prop1 in fields) {
            if (fields.hasOwnProperty(prop1)) {
                var cellid = recid + '-' + prop1;
                var value = instr[ii].data[prop1];
                if (value === undefined) {
                    value = '';
                }
                if (cmp.renderers[prop1] !== undefined) {

                    if (prop1 == 'e')
                        value = cmp.renderers[prop1].fn(cmp, instr[ii], cellid, value, value, prop1, instr[ii].data["realDate"], instr[ii].data["reccurenceType"]);
                    else
                        value = cmp.renderers[prop1].fn(cmp, instr[ii], cellid, value, value, prop1);
                } else {
                    value = '<span>' + value + '</span>';
                }
                var lc_ = '';
                if (lastField === prop1) {
                    lc_ = ' lastColCls';
                }
                jshtml += '<td  style="overflow: hidden; text-overflow: ellipsis;white-space: nowrap; width=100%" class="' + fields[prop1].TDCls + lc_ + '"><span id="' + cellid + '" class="' + fields[prop1].cellCls + '" >' + value + '</span></td>';
            }
        }
        jshtml += '</tr>';
    }
    jshtml += '</tbody>';
    jshtml += '</table>';
    if (endFrame !== undefined) {
        jshtml += endFrame;
    }
    if (tableConfig.renderTo === 'script') {
        return jshtml;
    }
    if (tableConfig.renderTo !== undefined) {
        var el = document.getElementById(tableConfig.renderTo);
        replaceHtml(el, jshtml);
        //el.innerHTML = jshtml;
    } else {
        window.document.write(jshtml);
    }
    $('input:hidden[name="totrows_counter"]')[0].value = instr.length.toString();

    applyFilterOnDateChanged();
    if (typeof documentSizeChanged == 'function') {
        documentSizeChanged();
    }
}

function getCellValueByKey(data, key, symbol) {
    var str = data[key];
    try {
        var newItem = str.replace(/"/g, '');
        newItem = newItem.replace("À€", "");

        if (newItem != '') {
            return newItem + ((symbol !== undefined) ? symbol : '');
        } else {
            return '';
        }
    } catch (ex) {
        return str;
    }
    return "";
}

function getRecordById(records, id) {
    for (var k = 0; k < records.length; k++) {
        if (records[k].id === id) {
            return records[k];
        }
    }
    return undefined;
}

function sortCmpGrid(v, field) {
    var sf = function (r1, r2) {
        if (r1.data.f25 < r2.data.f25) {
            return - 1;
        }
        if (r1.data.f25 > r2.data.f25) {
            return 1;
        }
        return 0;
    };
    return v.sort(sf);
}

function removeCallback(p) {
    var toRemove = - 1;
    for (var i = 0; i < cbfunctions.length; i++) {
        if (cbfunctions[i].cmp === p || (cbfunctions[i].cmp !== undefined && cbfunctions[i].cmp.id === p.id)) {
            toRemove = i;
        }
    }
    if (toRemove !== - 1) {
        cbfunctions.splice(toRemove, 1);
    }
}


function clearSearch() {
    var sw = document.getElementById('netd-window-search');
    sw.parentNode.removeChild(sw);
}

function clearSearchNews() {
    var swbody = document.getElementById('netd-testsearch');
    swbody.innerHTML = "";
    var sw = document.getElementById('netd-window-search-news');
    sw.style.visibility = 'hidden';
    sw.style.display = 'none';
    var news = document.getElementById('netd-newsGrid');
    news.style.visibility = "visible";
    news.style.display = "block";

}

function setSearchCombosVisible(visible) {
    var cb1 = document.getElementById('netd-search-type1');
    if (cb1) {
        cb1.style.visibility = "" + visible;
    }
    var cb2 = document.getElementById('netd-search-type2');
    if (cb2) {
        cb2.style.visibility = "" + visible;
    }
}


function isNumeric(n) {
    return !isNaN(parseFloat(n)) && isFinite(n);
}

function getDateSeconds(dateString) {
    var d = dateString.split('-');
    var date = new Date(d[0], d[1], d[2]);

    if (d.length > 3) {
        var sTime = d[3].split(':');
        date = new Date(d[0], d[1], d[2], sTime[0], sTime[1], sTime[2], 0);
    }
    return (date.getTime() / 1000.0);
}


function doSearchEnterPress(e) {
    if (e.keyCode == 13) {
        var btn = document.getElementById('searchButton');
        doSearch(btn);
        return false;
    }
}


function cbf_update_news(data, grid) {
    var hasNews = false;
    for (var i = 0, len = data.length; i < len; i++) {
        if (Netdania.Util.isInArray(data[i].i, grid.reqKeys)) {
            grid.reqKeys = $.grep(grid.reqKeys, function (value) {
                return value != data[i].i;
            });
            if (data[i].h) {
                var h, r;
                if (grid.tableConfig.dateAscending) {
                    for (var lj = 0; lj <= data[i].h.length - 1; lj++) {
                        hasNews = true;
                        h = data[i].h[lj];
                        var content = h.h.replace('{', '');
                        content = content.replace('}', '');
                        var contentArray = content.split(',');

                        contentArray = $.parseJSON(h.h);
                        //console.log(contentArray);
                        r = {
                            id: "recid" + lj,
                            data: {
                                t: h.t,
                                dt: h.t,
                                s: h.s,
                                i: h.i,
                                h: h.h,
                                c: getCellValueByKey(contentArray, "f5", ""),
                                e: getCellValueByKey(contentArray, "f1", ""),
                                im: getCellValueByKey(contentArray, "f3", ""),
                                a: getCellValueByKey(contentArray, "f7", getCellValueByKey(contentArray, "f9", "")),
                                f: getCellValueByKey(contentArray, "f8", getCellValueByKey(contentArray, "f9", "")),
                                p: getCellValueByKey(contentArray, "f6", getCellValueByKey(contentArray, "f9", "")),
                                currencyStr: getCellValueByKey(contentArray, "f5", ""),
                                parentID: getCellValueByKey(contentArray, "f10", ""),
                                realDate: getCellValueByKey(contentArray, "f13", ""),
                                reccurenceType: getCellValueByKey(contentArray, "f14", ""),
                                reccurencePeriod: getCellValueByKey(contentArray, "f15", ""),
                                provider: h.provider
                            }
                        };
                        grid.records.push(r);

                    }
                } else {
                    var length = data[i].h.length - 1;
                    var firstDate = undefined;
                    var index = - 1;
                    var intermediateIndex;
                    for (var lj = length; lj >= 0; lj--) {
                        hasNews = true;
                        h = data[i].h[lj];
                        index++;
                        intermediateIndex++;
                        var content = h.h.replace('{', '');
                        content = content.replace('}', '');
                        var contentArray = content.split(',');

                        contentArray = $.parseJSON(h.h);
                        //console.log(contentArray);
                        r = {
                            id: "recid" + lj,
                            data: {
                                t: h.t,
                                dt: h.t,
                                s: h.s,
                                i: h.i,
                                h: h.h,
                                c: getCellValueByKey(contentArray, "f5", ""),
                                e: getCellValueByKey(contentArray, "f1", ""),
                                im: getCellValueByKey(contentArray, "f3", ""),
                                a: getCellValueByKey(contentArray, "f7", getCellValueByKey(contentArray, "f9", "")),
                                f: getCellValueByKey(contentArray, "f8", getCellValueByKey(contentArray, "f9", "")),
                                p: getCellValueByKey(contentArray, "f6", getCellValueByKey(contentArray, "f9", "")),
                                currencyStr: getCellValueByKey(contentArray, "f5", ""),
                                parentID: getCellValueByKey(contentArray, "f10", ""),
                                realDate: getCellValueByKey(contentArray, "f13", ""),
                                reccurenceType: getCellValueByKey(contentArray, "f14", ""),
                                reccurencePeriod: getCellValueByKey(contentArray, "f15", ""),
                                provider: h.provider
                            }
                        };

                        var dateNews = new Date(h.t * 1000);
                        if (firstDate == undefined || dateFormat(dateNews, dateFormat.masks.shortDate) != firstDate) {
                            firstDate = dateFormat(dateNews, dateFormat.masks.shortDate);
                            intermediateIndex = index;
                            grid.records.splice(index, 0, r)
                        } else {
                            intermediateIndex--;
                            grid.records.splice(intermediateIndex, 0, r)
                        }
                    }
                }
                if (hasNews === false) {
                    hasNews = true;
                    h = data[i].h;

                    r = {
                        id: "recid" + lj,
                        data: {
                            t: h.t,
                            s: h.s,
                            i: h.i,
                            h: h.h
                        }
                    };
                    grid.records.splice(0, 0, r);
                    grid.records.splice(grid.records.length - 1, 1);
                }
            }

        }
    }
    if (hasNews === true || grid.newsPainted !== true) {
        grid.newsPainted = true;

        if (grid.type === 'newslookup') {
            var tdcid = queryString("TDC_ID");
            tdcid = (tdcid !== "false") ? ("&TDC_ID=" + tdcid) : '';
            grid.reqKeys = [];
        }

        renderTable(grid.tableConfig);
    }

}


var fromTimer = false;
function Refresh(renderTo, groupByDate, dateMode, dateAscending, tableConfig) {
    tid = setTimeout('refreshAfterTimer(\'' + renderTo + '\',' + groupByDate + ',\'' + dateMode + '\',' + dateAscending + '\',' + tableConfig + ')', 180000);
}

function refreshAfterTimer(renderTo, groupByDate, dateMode, dateAscending, tableConfig) {
    fromTimer = true;
    renderNews(renderTo, groupByDate, dateMode, false, dateAscending, tableConfig);
}


function setCookie(c_name, value, exdays) {
    var exdate = new Date();
    exdate.setDate(exdate.getDate() + exdays);
    var c_value = escape(value) + ((exdays == null) ? "" : "; expires=" + exdate.toUTCString());
    document.cookie = c_name + "=" + c_value;
}

function getCookie(c_name) {
    var i, x, y, ARRcookies = document.cookie.split(";");
    for (i = 0; i < ARRcookies.length; i++) {
        x = ARRcookies[i].substr(0, ARRcookies[i].indexOf("="));
        y = ARRcookies[i].substr(ARRcookies[i].indexOf("=") + 1);
        x = x.replace(/^\s+|\s+$/g, "");
        if (x == c_name) {
            return unescape(y);
        }
    }
}
function SetFromTimer() {
    setCookie("DateFromNetStation", GetWeekReference(), 1);
}

function GetWeekReference() {
    var weekReference;
    if ($('input:hidden[name="week"]')[0].value != "" && $('input:hidden[name="week"]')[0].value != undefined) {
        var datestart = $('input:hidden[name="week"]')[0];
        weekReference = datestart.value;
    } else {
        var date = new Date();
        $('input:hidden[name="week"]')[0].value = date.getFullYear() + "-" + date.getMonth() + "-" + date.getDate() + " 00:00:00";
        weekReference = $('input:hidden[name="week"]')[0].value;
    }
    return weekReference;
}

function delete_cookie(name) {
    document.cookie = name + '=; expires=Thu, 01-Jan-70 00:00:01 GMT;';
}

function renderNews(renderTo, groupByDate, dateMode, moveNext, dateAscending, tableConfig) {

    var host = Netdania.General.Config.Host; //the NetDania server to connect to
    connectionConfig = {
        pollingInterval: Netdania.JsApi.PoolingInterval.AUTO,
        //milliseconds
        host: host,
        behavior: Netdania.JsApi.ConnectionType.AUTO,
        //LONG POLLING
        type: 1 //the server response format: JSONP
    }

    connection = new Netdania.JsApi.JSONConnection(connectionConfig);
    //connection.Connect();

    var previousDate;
    tid = setTimeout('refreshAfterTimer(\'' + renderTo + '\',' + groupByDate + ',\'' + dateMode + '\',' + dateAscending + '\',' + tableConfig + ')', 180000);
    clearTimeout(tid);

    Refresh(renderTo, groupByDate, dateMode, dateAscending);
    window['netd-news-' + globalCurrentCmp] = {
        id: 'netd-news-' + globalCurrentCmp,
        current: "" + globalCurrentCmp,
        renderers: {
            dt: {
                fn: function (grid, rec, cellid, val, oldval, fxy) {
                    var cd = new Date(val * 1000);
                    if (dateFormat(cd, "dddd, mmmm dd") != previousDate) {
                        previousDate = dateFormat(cd, "dddd, mmmm dd");
                        //return '<div class=\'e-cal-date\'><span>'+dateFormat(cd, "dddd")+'<br />'+dateFormat(cd, "mmmm dd")+'</span></div>';
                        return '<div class=\'e-cal-date\'><span>' + dateFormat(cd, "dd-mm-yyyy") + '</span></div>';
                    } else {
                        //return '<div class=\'e-cal-date\' style="display:none;"><span>'+dateFormat(cd, "dddd")+'<br />'+dateFormat(cd, "mmmm dd")+'</span></div>';
                        return '<div class=\'e-cal-date\' style="display:none;"><span>' + dateFormat(cd, "dd-mm-yyyy") + '</span></div>';
                    }
                }
            },
            c: {
                fn: function (grid, rec, cellid, val, oldval, fxy) {
                    var storyCls = '';
                    if (rec.data.s != 1) {
                        storyCls = 'class="missingStory"';
                    }
                    if (val == "")
                        return "";
                    return '<img width="20px" height="20px" alt="' + val + '" title="' + val + '" src="/economiccalendar/tradable/images/Currency/' + val + '.png?v=1.0" ></img>';
                }
            },
            e: {
                fn: function (grid, rec, cellid, val, oldval, fxy, realDate, reccurence) {
                    var storyCls = '';

                    var realDateMonth = '';

                    if (realDate !== undefined) {
                        if (reccurence == "week") {
                            var dateUnf = new Date(realDate * 1000);
                            if (dateUnf.getFullYear() == new Date().getFullYear())
                                realDateMonth = '(' + dateFormat(dateUnf, "mmm dd").toUpperCase() + ')';
                            else
                                realDateMonth = '(' + dateFormat(dateUnf, "mmm dd, yyyy").toUpperCase() + ')';
                        } else {
                            var dateUnf = new Date(realDate * 1000);
                            if (dateUnf.getFullYear() == new Date().getFullYear())
                                realDateMonth = '(' + dateFormat(dateUnf, "mmm").toUpperCase() + ')';
                            else
                                realDateMonth = '(' + dateFormat(dateUnf, "mmm, yyyy").toUpperCase() + ')';
                        }
                    }
                    return '<span title="' + val + realDateMonth + '">' + val + realDateMonth + '</span>';
                }
            },
            im: {
                fn: function (grid, rec, cellid, val, oldval, fxy) {
                    return '<table><tr><td class=\'evImportance ' + val.toLowerCase() + '\' style=\'border: 0px solid transparent;\'>' + val + '</td></tr></table>';

                }
            },
            t: {
                fn: function (grid, rec, cellid, val, oldval, fxy) {
                    if (val === '' || val === "N/A" || val === undefined) {
                        return "";
                    }
                    var dateUnf = new Date(val * 1000);
                    return '<span>' + dateFormat(dateUnf, "HH:MM ").toLowerCase() + '</span>';
                }
            },
            currencyStr: {
                fn: function (grid, rec, cellid, val, oldval, fxy) {
                    if (val === '' || val === "N/A" || val === undefined) {
                        return "";
                    }
                    return '<span style="display:none;">' + val + '</span>';
                }
            },
            //        realDate: { fn: function (grid, rec, cellid, val, oldval, fxy) {
            //            if (val === '' || val === "N/A" || val === undefined) {
            //                return "";
            //            }
            //            var dateUnf = new Date(val * 1000);
            //            return '<span>' + dateFormat(dateUnf, "mmm").toUpperCase() + '</span>';
            //        }
            //        },
            d: {
                fn: function (grid, rec, cellid, val, oldval, fxy) {
                    var storyCls = '';
                    if (rec.data.s != 1) {
                        return "";
                    } else {
                        return '<img style="cursor: pointer;" src="images/arrow_state_grey_expanded.png" onmouseover="this.src=\'images/arrow_state_blue_expanded.png\'"' +
                        'onmouseout="this.src=\'images/arrow_state_grey_expanded.png\'" ' + storyCls + '></img>';
                    }
                }
            }

        },
        records: [
        ],
        columns: {
            t: 0,
            c: 0,
            e: 0,
            i: 0 
        }
    };
    var cmp = window['netd-news-' + globalCurrentCmp];

    var fields;
    if (tableConfig !== undefined && tableConfig.columns !== undefined) {
        fields = tableConfig.columns;
    } else {
        fields = {
            dt: {
                name: 'Date',
                value: '',
                width: 75,
                hdTDcls: 'TDClsCenter',
                TDCls: 'TDClsCenter',
                cellCls: 'kursColumn',
                hdCellCls: 'gridDefaultHeader' 
            },
            t: {
                name: 'Time',
                value: '',
                width: 35,
                hdTDcls: 'TDClsCenter',
                TDCls: 'TDClsCenter',
                cellCls: 'kursColumn',
                hdCellCls: 'gridDefaultHeader' 
            },
            c: {
                name: 'Currency',
                value: '',
                width: 60,
                hdTDcls: 'TDClsCenter',
                TDCls: 'TDClsCenter',
                cellCls: 'nameColumn',
                hdCellCls: 'nameHeaderClass' 
            },
            e: {
                name: 'Event',
                value: '',
                width: 200,
                hdTDcls: 'TDClsCenter',
                TDCls: 'TDClsLeft',
                cellCls: 'nameColumn',
                hdCellCls: 'nameHeaderClass' 
            },
            im: {
                name: 'Importance',
                value: '',
                width: 70,
                hdTDcls: 'TDClsCenter',
                TDCls: 'TDClsCenter',
                cellCls: 'TDClsImportance',
                hdCellCls: 'hdTDClsImportance' 
            },
            a: {
                name: 'Actual',
                value: '',
                width: 70,
                hdTDcls: 'hdTDClsCenter',
                TDCls: 'TDClsCenter',
                cellCls: 'nameColumnActual',
                hdCellCls: 'nameHeaderClass' 
            },
            f: {
                name: 'Forecast',
                value: '',
                width: 70,
                hdTDcls: 'hdTDClsCenter',
                TDCls: 'TDClsCenter',
                cellCls: 'nameColumn',
                hdCellCls: 'nameHeaderClass' 
            },
            p: {
                name: 'Previous',
                value: '',
                width: 70,
                hdTDcls: 'hdTDClsCenter',
                TDCls: 'TDClsCenter',
                cellCls: 'nameColumn',
                hdCellCls: 'nameHeaderClass' 
            },
            d: {
                name: 'Note',
                value: '',
                width: 30,
                hdTDcls: 'hdTDClsLeft',
                TDCls: 'TDClsLeft',
                cellCls: 'nameColumn',
                hdCellCls: 'nameHeaderClass' 
            },
            currencyStr: {
                name: 'currencyStr',
                value: '',
                width: 1,
                hdTDcls: 'currencyStr',
                TDCls: 'currencyStr',
                cellCls: 'currencyStr',
                hdCellCls: 'currencyStr' 
            }
        };
    }

    //var strSource ="RBB_StockExchange"; var intMax= 250; strProvider = "rb-borsen"; var boolMonitor = true;
    var strSource = "EconomicalCalendar";
    var intMax = 250;
    if (renderTo == 'netd-newsGrid-homepage') {
        intMax = 30;
    }
    strProvider = "netdania_calendar";
    var boolMonitor = true;
    var tdcid = queryString("TDC_ID");
    tdcid = (tdcid !== "false") ? ("&TDC_ID=" + tdcid) : '';
    cmp.tableConfig = {
        instr: cmp,
        provider: strProvider,
        fields: fields,
        contentClass: (tableConfig !== undefined && tableConfig.contentClass !== undefined) ? tableConfig.contentClass : '',
        contentOddClass: (tableConfig !== undefined && tableConfig.contentOddClass !== undefined) ? tableConfig.contentOddClass : '',
        renderTo: renderTo,
        groupByDate: (groupByDate === true) ? true : false,
        hideHeaders: (tableConfig !== undefined && tableConfig.hideHeader !== undefined) ? tableConfig.hideHeader : false,
        dateAscending: dateAscending,
        click: (tableConfig !== undefined && tableConfig.click !== undefined) ? tableConfig.click : true
    };
    cbfunctions = new Array();






    cbfunctions.push({
        update: cbf_update_news,
        cmp: cmp
    });


    connection.addListener(Netdania.JsApi.Events.ONRAWUPDATE, function (data, ts, id, type) {
        for (var kk = 0; kk < cbfunctions.length; kk++) {

            if (cbfunctions[kk].hasOwnProperty('win')) {
                if (cbfunctions[kk].hasOwnProperty('win') !== null) {
                    cbfunctions[kk].update(data, cbfunctions[kk].cmp, cbfunctions[kk].win);
                }
            } else {
                cbfunctions[kk].update(data, cbfunctions[kk].cmp);
            }

        }
    });



    //    cmp.noDataMessage = 'There are no events scheduled!';
    cmp.noDataMessage = "";
    var d1 = "";
    var d2 = "";
    var weekReference;
    if ($('input:hidden[name="week"]')[0].value != "" && $('input:hidden[name="week"]')[0].value != undefined) {
        var datestart = $('input:hidden[name="week"]')[0];
        weekReference = datestart.value;
    } else {
        var date = new Date();
        $('input:hidden[name="week"]')[0].value = date.getFullYear() + "-" + date.getMonth() + "-" + date.getDate() + " 00:00:00";
        weekReference = $('input:hidden[name="week"]')[0].value;
    }
    openedStories = new Array();

    var color = theme == 'black' ? 'white' : 'black';
    var alternateColor = 'gray';

    var dateFromCookie = getCookie("DateFromNetStation");
    if (dateFromCookie !== undefined && dateFromCookie !== "" && !moveNext) {
        weekReference = dateFromCookie;
        delete_cookie('DateFromNetStation');
        fromTimer = true;
    }
    if (fromTimer) {
        var curr = getDate(weekReference); // get current date 
        var first = curr.getDate() - curr.getDay(); // First day is the day of the month - the day of the week 
        var firstday = new Date(curr.setDate(first));
        $('input:hidden[name="week"]')[0].value = firstday.getFullYear() + "-" + firstday.getMonth() + "-" + firstday.getDate() + " 00:00:00";
        var lastday = new Date(firstday);
        lastday.setDate(firstday.getDate() + 6);

        d1 = getDateSeconds(firstday.getFullYear() + "-" + firstday.getMonth() + "-" + firstday.getDate());
        d2 = getDateSeconds(lastday.getFullYear() + "-" + lastday.getMonth() + "-" + lastday.getDate() + "- 23:59:59");
        SetFromTimer();
    } else {
        switch (dateMode) {
        case ("today"):
            {
                var date = new Date();
                //            alert(date);
                d1 = getDateSeconds(date.getFullYear() + "-" + date.getMonth() + "-" + date.getDate() + "- 00:00:00");
                d2 = getDateSeconds(date.getFullYear() + "-" + date.getMonth() + "-" + date.getDate() + "- 23:59:59");

                $('#dateInterval').text(" on " + date.getFullYear() + "-" + (parseInt(date.getMonth().toString()) + 1) + "-" + date.getDate());
                break;
            }
        case ("current"):
            {
                var curr = new Date(); // get current date 
                var first = curr.getDate() - curr.getDay(); // First day is the day of the month - the day of the week 
                var firstday = new Date(curr.setDate(first));
                var lastday = new Date(firstday);
                lastday.setDate(firstday.getDate() + 6);
                $('input:hidden[name="week"]')[0].value = firstday.getFullYear() + "-" + firstday.getMonth() + "-" + firstday.getDate() + " 00:00:00";
                d1 = getDateSeconds(firstday.getFullYear() + "-" + firstday.getMonth() + "-" + firstday.getDate());
                d2 = getDateSeconds(lastday.getFullYear() + "-" + lastday.getMonth() + "-" + lastday.getDate() + "- 23:59:59");
                $('#dateInterval').text(" between " + firstday.getFullYear() + "-" + (parseInt(firstday.getMonth().toString()) + 1) + "-" + firstday.getDate() + " and " + lastday.getFullYear() + "-" + (lastday.getMonth() + 1) + "-" + lastday.getDate());
                $('span[id="currentWeekSpan"]')[0].style.fontWeight = 'bold';
                $('span[id="previousWeekSpan"]')[0].style.fontWeight = '';
                $('span[id="nextWeekSpan"]')[0].style.fontWeight = '';
                $('span[id="currentWeekSpan"]')[0].style.color = color;
                $('span[id="previousWeekSpan"]')[0].style.color = alternateColor;
                $('span[id="nextWeekSpan"]')[0].style.color = alternateColor;
                SetFromTimer();
                break;

            }
        case ("previous"):
            {
                var curr = getDate(weekReference); // get current date 
                var first = curr.getDate() - curr.getDay() - 7; // First day is the day of the month - the day of the week 
                var firstday = new Date(curr.setDate(first));
                $('input:hidden[name="week"]')[0].value = firstday.getFullYear() + "-" + firstday.getMonth() + "-" + firstday.getDate() + " 00:00:00";
                var lastday = new Date(firstday);
                lastday.setDate(firstday.getDate() + 6);

                d1 = getDateSeconds(firstday.getFullYear() + "-" + firstday.getMonth() + "-" + firstday.getDate());
                d2 = getDateSeconds(lastday.getFullYear() + "-" + lastday.getMonth() + "-" + lastday.getDate() + "- 23:59:59");
                $('#dateInterval').text(" between " + firstday.getFullYear() + "-" + (parseInt(firstday.getMonth().toString()) + 1) + "-" + firstday.getDate() + " and " + lastday.getFullYear() + "-" + (lastday.getMonth() + 1) + "-" + lastday.getDate());
                $('span[id="currentWeekSpan"]')[0].style.fontWeight = '';
                $('span[id="previousWeekSpan"]')[0].style.fontWeight = 'bold';
                $('span[id="nextWeekSpan"]')[0].style.fontWeight = '';
                $('span[id="currentWeekSpan"]')[0].style.color = alternateColor;
                $('span[id="previousWeekSpan"]')[0].style.color = color;
                $('span[id="nextWeekSpan"]')[0].style.color = alternateColor;
                SetFromTimer();
                break;

            }
        case ("next"):
            {
                var curr = getDate(weekReference); // get current date 
                var first = curr.getDate() - curr.getDay() + 7; // First day is the day of the month - the day of the week 
                var firstday = new Date(curr.setDate(first));
                $('input:hidden[name="week"]')[0].value = firstday.getFullYear() + "-" + firstday.getMonth() + "-" + firstday.getDate() + " 00:00:00";
                var lastday = new Date(firstday);
                lastday.setDate(firstday.getDate() + 6);
                d1 = getDateSeconds(firstday.getFullYear() + "-" + firstday.getMonth() + "-" + firstday.getDate());
                d2 = getDateSeconds(lastday.getFullYear() + "-" + lastday.getMonth() + "-" + lastday.getDate() + "- 23:59:59");
                $('#dateInterval').text(" between " + firstday.getFullYear() + "-" + (parseInt(firstday.getMonth().toString()) + 1) + "-" + firstday.getDate() + " and " + lastday.getFullYear() + "-" + (lastday.getMonth() + 1) + "-" + lastday.getDate());
                $('span[id="currentWeekSpan"]')[0].style.fontWeight = '';
                $('span[id="previousWeekSpan"]')[0].style.fontWeight = '';
                $('span[id="nextWeekSpan"]')[0].style.fontWeight = 'bold';
                $('span[id="currentWeekSpan"]')[0].style.color = alternateColor;
                $('span[id="previousWeekSpan"]')[0].style.color = alternateColor;
                $('span[id="nextWeekSpan"]')[0].style.color = color;
                SetFromTimer();
                break;

            }
        }
    }


    var byteSearchIn;
    if (byteSearchIn === undefined) {
        byteSearchIn = 0;
    }

    var rec = Netdania.JsApi.Request.getReqObjNewsSearch(strSource, '', intMax, d1, d2, byteSearchIn, strProvider);
    var recid = 'recid-' + rec.i;
    cmp.reqKeys = [rec.i];

    connection.AddRequest(rec);
    connection.Flush();

}


function daysInMonth(iMonth, iYear) {
    return new Date(iYear, iMonth, 0).getDate();
}

function PageQuery(q) {
    if (q.length > 1) {
        this.q = q.substring(1, q.length);
    } else {
        this.q = null;
    }
    this.keyValuePairs = [];
    if (q) {

        for (var i = 0; i < this.q.split("&").length; i++) {
            this.keyValuePairs[i] = this.q.split("&")[i];
        }
    }

    this.getKeyValuePairs = function () {
        return this.keyValuePairs;
    };

    this.getValue = function (s) {
        for (var j = 0; j < this.keyValuePairs.length; j++) {
            if (this.keyValuePairs[j].split("=")[0] == s) {
                return this.keyValuePairs[j].split("=")[1];
            }
        }
        return false;
    };

    this.getParameters = function () {

        var a = new Array(this.getLength());

        for (var j = 0; j < this.keyValuePairs.length; j++) {
            a[j] = this.keyValuePairs[j].split("=")[0];
        }
        return a;
    };

    this.getLength = function () {
        return this.keyValuePairs.length;
    };
}
function queryString(key) {
    var page = new PageQuery(window.location.search);
    return unescape(page.getValue(key));
}


function SetChecked(val, chkName) {
    var checkBoxes = document.getElementsByName(chkName);
    for (var i = 0; i < checkBoxes.length; i++) {
        checkBoxes[i].checked = val;
    }
    var omxc20 = document.getElementById('netd-OMXC20');
    omxc20.checked = true;
}

function createwin(headline, date) {
    var ct = document.createElement('div');
    ct.setAttribute("id", "netd-news-story-mask");

    var story = document.createElement('div');
    story.setAttribute("id", "netd-news-story-body");
    story.style.height = "600px";

    var quote = document.getElementById("netd-quote-body");
    if (quote) {
        ct.style.height = "805px";
        quote.appendChild(ct);
        quote.appendChild(story);
    } else {
        var body = document.getElementById("netd-fq-body");
        ct.style.height = "785px";
        body.appendChild(ct);
        body.appendChild(story);
    }
    var dateUnf = new Date(date * 1000);
    var d = dateFormat(dateUnf, "dd/mm HH:MM").toLowerCase();
    //	story.innerHTML = '<div class="netd-window-search-hd">'+
    //		'<div class="netd-window-headline-title" id="netd-headline-bar">'+stripHeadline(unescape(headline),80)+'</div>'+
    //		'<div class="netd-btnClose" onclick="clearStory();"> X</div></div>'+
    //		'<div id="netd-teststory"><h3>'+unescape(headline)+'</h3><h3>'+ d +'</h3><div id="netd-teststory-body">Henter data...</div></div>'+
    //		'<div class="netd-window-search-buttons"><div class="netd-btnSubmitPlain" onclick="clearStory();"><a >X</a></div></div>';
}

function getStory(xdata, reqId, win) {
    for (var i = 0, len = xdata.length; i < len; i++) {
        if (xdata[i].i === reqId) {
            if (xdata[i].s !== null && xdata[i].s != '') {
                var storyObj = jQuery.parseJSON(xdata[i].s);

                if (storyObj !== null && storyObj.f11 != undefined && storyObj.f11.length > 0) {

                    var line1 = new Array();
                    var dates = new Array();
                    var historyValues = new Array();
                    var maxY = 0;
                    var stepWidth = 0;

                    var realEventDateExistsForAllChildren = true;
                    for (var i = 0; i < storyObj.f11.length; i++) {
                        if (storyObj.f11[i].f13 == undefined)
                            realEventDateExistsForAllChildren = false;
                    }

                    for (var i = 0; i < storyObj.f11.length; i++) {
                        var dateTimeEvent = new Date(storyObj.f11[i].f12 * 1000);

                        if (realEventDateExistsForAllChildren) {
                            dateTimeEvent = new Date(storyObj.f11[i].f13 * 1000);
                        }

                        var date = dateTimeEvent.getFullYear() + "-" + (parseInt(dateTimeEvent.getMonth().toString()) + 1) + "-" + dateTimeEvent.getDate();
                        var tempDate = dateTimeEvent;


                        dates.push(date.toString());

                        //cristiancr- in IE8 we receive an error when trying to render an missing point
                        if (!(!storyObj.f11[i].f7)) {
                            line1.push([date.toString(), parseFloat(storyObj.f11[i].f7)]);
                            if (parseFloat(storyObj.f11[i].f7) > maxY) {
                                stepWidth = parseFloat(storyObj.f11[i].f7) - maxY;
                                maxY = parseFloat(storyObj.f11[i].f7);
                            }
                        }
                        historyValues.push([date.toString()
                        , ((storyObj.f11[i].f7 !== undefined) ? storyObj.f11[i].f7 + ((storyObj.f11[i].f9 !== undefined) ? storyObj.f11[i].f9 : '') : '')
                        , ((storyObj.f11[i].f8 !== undefined) ? storyObj.f11[i].f8 + ((storyObj.f11[i].f9 !== undefined) ? storyObj.f11[i].f9 : '') : '')
                        , ((storyObj.f11[i].f6 !== undefined) ? storyObj.f11[i].f6 + ((storyObj.f11[i].f9 !== undefined) ? storyObj.f11[i].f9 : '') : '')]);

                    }
                    var symbol = getCellValueByKey(storyObj.f11[0], "f9", "");
                    if (symbol === undefined) {
                        symbol = '';
                    }
                    var noOfTicks = line1.length;
                    var gridProp = {
                        background: 'rgba(57,57,57,0.0)',
                        gridLineColor: '#666666',
                        shadow: false 
                    };
                    var seriesColor = theme == 'black' ? 'rgb(255, 102, 0)' : '#0096D5';
                    //cristiancr for one point we get an error in IE8 on rendering chart
                    if (line1.length > 1) {
                        $('div[id="chartDiv' + lastLoadedChartID + '"]')[0].innerHTML = '';
                        plot1 = $.jqplot("chartDiv" + lastLoadedChartID.toString(), [line1], {
                            animate: true,
                            // Will animate plot on calls to plot1.replot({resetAxes:true})        
                            animateReplot: true,

                            series: [
                            {
                                lineWidth: '1',
                                color: seriesColor,
                                markerOptions: {
                                    style: 'circle',
                                    size: 5,
                                    color: '#ED1C24'
                                }
                            }
                            ],
                            axesDefaults: {
                                tickRenderer: $.jqplot.CanvasAxisTickRenderer,
                                tickOptions: {
                                    fontFamily: 'Verdana',
                                    fontSize: '7pt'
                                }
                            },
                            cursor: {
                                show: false
                            },

                            highlighter: {
                                show: true,
                                sizeAdjust: 7.5,
                                tooltipLocation: 's'
                            },
                            grid: gridProp,
                            axes: {
                                // These options will set up the x axis like a category axis.            
                                xaxis: {
                                    pad: 1.2,
                                    label: " ",
                                    renderer: $.jqplot.DateAxisRenderer,
                                    rendererOptions: {
                                        tickRenderer: $.jqplot.CanvasAxisTickRenderer 
                                    },
                                    tickOptions: {
                                        angle: - 90,
                                        formatString: '%d/%m/%y',
                                        showGridline: false 
                                    },
                                    min: line1[0][0],
                                    autoscale: true,
                                    //numberTicks:noOfTicks+2,
                                    ticks: dates
                                },
                                yaxis: {
                                    label: 'Actual value',
                                    //max:maxY+stepWidth,
                                    //numberTicks:noOfTicks+1,
                                    labelRenderer: $.jqplot.CanvasAxisLabelRenderer,
                                    labelOptions: {
                                        angle: - 90,
                                        fontFamily: 'Verdana',
                                        fontSize: '8pt' 
                                    },
                                    rendererOptions: {
                                        tickRenderer: $.jqplot.CanvasAxisTickRenderer 
                                    },
                                    tickOptions: {
                                        formatString: '%.2f ' + symbol 
                                    }
                                }
                            }
                        });
                    } else {
                        $('div[id="chartDiv' + lastLoadedChartID + '"]')[0].innerHTML = '<span>Not enough history data available.</span>';

                    }
                    FillHistory(historyValues);
                    if (controlDivToHide != undefined) {
                        controlDivToHide.style.display = "none";
                    }
                    removeCallback(reqId);

                } else {
                    var storyContent = $("<div/>").html(storyObj.f2).text();
                    if (document.getElementById("netd-teststory-bodystory_" + reqId) !== null) {
                        document.getElementById("netd-teststory-bodystory_" + reqId).innerHTML = storyContent;
                        //removeCallback(reqId);
                        return;
                    }
                }
                if (document.getElementById('netd-newsGrid') !== null) {
                    //storyContent+='<div class="netd-window-search-buttons"><div class="netd-btnCloseStory" onclick="collapseStory(this);"><a >Luk</a></div></div>';
                    if (document.getElementById("story_" + reqId) !== null) {
                        document.getElementById("story_" + reqId).innerHTML = storyContent;
                    }

                }
                removeCallback(reqId);
                //removeArrayElement(cbfunctions, 'getStory');
            }
        }
    }
}

// This function inserts newNode after referenceNode
function insertAfter(referenceNode, newNode) {
    referenceNode.parentNode.insertBefore(newNode, referenceNode.nextSibling);
}

function creatediv(id, span, html, title, actual, previous, forecast, currency, storyId, provider, parentID, width, height, left, top) {

    var newdiv = document.createElement('div');
    newdiv.setAttribute('id', id);

    if (width) {
        newdiv.style.width = 300;
    }

    if (height) {
        newdiv.style.height = 300;
    }

    if ((left || top) || (left && top)) {
        newdiv.style.position = "absolute";

        if (left) {
            newdiv.style.left = left;
        }

        if (top) {
            newdiv.style.top = top;
        }
    }

    newdiv.style.border = "0px solid #000";
    newdiv.style.width = "450px";

    if (html) {
        newdiv.innerHTML = html;
    } else {
        newdiv.innerHTML = "nothing";
    }
    var trNode = span;
    addRow(title, newdiv, trNode, actual, previous, forecast, currency, id, storyId, provider, parentID);
}

function addRow(title, content, trNode, actual, previous, forecast, currency, id, storyId, provider, parentID) {
    row = document.createElement("TR");
    row.setAttribute("class", "expanded");
    cell1 = document.createElement("TD");
    cell1.setAttribute("class", "storyStyle");
    cell1.setAttribute("id", 'storyStyle' + storyId);
    cell1.colSpan = "9";

    textnode1 = document.createTextNode('');

    cell1.appendChild(textnode1);
    var newTable = "<table id='styleTable'><tr><td style='width:40px;'><img src='images/Currency/|CurrencyFlag.png?v=1.0' style='width:24px; height:24px; margin-left:10px;' /></td> <td class='titleStyle' colspan='9'><span>|Title</span></td></tr></table>";
    newTable += "<tr><td><br /></td></tr><tr><td class='trend' colspan='3'><table class='tableTrend'><tr><td style='width:50px;'>|StockIndexImage</td>";
    var valuesEvolution = " <td class='trend'><table style='width:140px; margin-left:10px;  border-bottom: 1px solid transparent;'><tr><td class='titleStyle'><span>Actual</span></td><td class='titleStyle' style='color:|ActualColor'><span>|ActualValue</span></td></tr><tr><td ><span>Previous</span></td><td ><span>|PreviousValue</span></td></tr><tr><td ><span>Forecast</span></td><td ><span>|ForecastValue</span></td></tr></table></td>";
    var contentHTML = " <td colspan='7' id='netd-teststory-body" + id + "' style='padding-left:15px; padding-right:5px; text-align: justify;'></td></tr></table><br /></td>";
    var historyHTML = "<tr><td colspan='9'><table id='HistoryTable'><tr><td class='inactiveTab' id='Chart|StoryID' onclick=\"javascript:ChangeHistoryTab('Chart|StoryID',|StoryID,'|Provider',|ParentID);\"><span>Chart</span></td><td class='inactiveTab' id='History|StoryID' onclick=\"javascript:ChangeHistoryTab('History|StoryID',|StoryID,'|Provider',|ParentID);\"><span>History</span></td><td class='fillTab'><span><div style='width:100%''></div></span></td></tr><tr><td id='Body|StoryID' colspan='4' class='tabBody'><div id='chartDiv|StoryID' ></div><div id='historyDiv|StoryID' style='display:none; vertical-align:top;'></div></td></tr></table></td></tr>";

    newTable = newTable.replace("|CurrencyFlag", currency);
    newTable = newTable.replace("|Title", unescape(title));

    var trend = "0";
    var color = theme == 'White' ? "black" : "white";

    if (actual != "" && previous != "") {
        if (parseFloat(actual) > parseFloat(previous)) {
            trend = "1";
            color = theme == 'White' ? "green" : "lime";
        } else if (parseFloat(actual) < parseFloat(previous)) {
            trend = "-1";
            color = theme == 'White' ? "red" : "orange";
        }
    }
    var imageTrend = "";
    switch (trend) {
    case "1":
        imageTrend = "<img src='images/StockIndexUp.png' style='margin-left:10px' > </img>";
        break;
    case "-1":
        imageTrend = "<img src='images/StockIndexDown.png' style='margin-left:10px' > </img>";
        break;
    }
    newTable = newTable.replace("|StockIndexImage", imageTrend);

    if (actual != "" || previous != "" || forecast != "" ) {
        valuesEvolution = valuesEvolution.replace("|ActualColor", color);
        valuesEvolution = valuesEvolution.replace("|ActualValue", ((actual == "undefined") ? "N/A" : actual));
        valuesEvolution = valuesEvolution.replace("|PreviousValue", ((previous == "undefined") ? "N/A" : previous));
        valuesEvolution = valuesEvolution.replace("|ForecastValue", ((forecast == "undefined") ? "N/A" : forecast));
        newTable += valuesEvolution;
    } else {
        newTable += "<td></td>";
    }

    newTable += contentHTML;

    if (parentID == "" && storyId != "") {
        parentID = storyId;
    }
    historyHTML = historyHTML.replace(/\|StoryID/g, storyId);
    historyHTML = historyHTML.replace(/\|Provider/g, provider);
    historyHTML = historyHTML.replace(/\|ParentID/g, parentID);
    newTable += historyHTML;


    cell1.innerHTML = newTable;
    row.appendChild(cell1);
    insertAfter(trNode, row);
    //scroll into view
    //$('td[id$=storyStyle' + storyId + ']').scrollIntoView();
    //end scroll
}


function showFilter() {

    var btnFilter = document.getElementById("filterMenu");
    if (btnFilter != undefined) {
        if (btnFilter.style.display == 'none') {
            btnFilter.style.display = 'block';
        } else {
            btnFilter.style.display = 'none';
        }
    }
}

function showStory(span, title, storyId, provider, actual, previous, forecast, currency, parentID, headline, date) {


    var storyReq;
    if (headline !== undefined) {
        createwin(headline, date);

        storyReq = getReqObjStory(storyId, provider, false);



        cbfunctions.push({
            update: getStory,
            cmp: storyReq.i
        });


        connection.AddRequest(storyReq);
        connection.Flush();
        return;
    }
    var trNode;
    if (span.className === 'storyVisible') {
        trNode = span;
        span.className = "storyHidden";
        trNode.nextSibling.style.display = 'none';
        for (var elem = 0; elem < span.childNodes.length; elem += 1) {
            if (span.childNodes[elem] != undefined) {
                span.childNodes[elem].style.borderBottomStyle = "solid";
            }
        }

        controlDivToHide = $('div[id="' + storyId + '"]')[0];

        if ($('div[id="chartDiv' + storyId + '"]').length > 0) {
            //cristiancr  -always show the chart or history
            //    $('div[id="chartDiv' + storyId + '"]')[0].style.display = "none";
        }
        if ($('div[id="historyDiv' + storyId + '"]').length > 0) {
            //cristiancr -always show the chart or history
            //  $('div[id="historyDiv' + storyId + '"]')[0].style.display = "none";
        }
    } else if (span.className === "storyHidden") {
        trNode = span;
        span.className = "storyVisible";
        trNode.nextSibling.style.display = '';
        for (var elem = 0; elem < span.childNodes.length; elem += 1) {
            if (span.childNodes[elem] != undefined) {
                span.childNodes[elem].style.borderBottomStyle = "none";
            }
        }
    } else {
        storyReq = getReqObjStory(storyId, provider, false);

        cbfunctions.push({
            update: getStory,
            cmp: storyReq.i
        });



        connection.AddRequest(storyReq);
        connection.Flush();

        var objRequest = jQuery.parseJSON(storyReq.s);
        span.className = "storyVisible";
        for (var elem = 0; elem < span.childNodes.length; elem += 1) {
            if (span.childNodes[elem] != undefined) {
                span.childNodes[elem].style.borderBottomStyle = "none";
            }
        }
        creatediv('story_' + storyReq.i, span, 'Loading data...', title, actual, previous, forecast, currency, objRequest.StoryId, provider, parentID);
        document.getElementById("netd-teststory-bodystory_" + storyReq.i).innerHTML = '<img valign="middle" src="/jscomponents/cssgeneral/images/loading.gif"/>';

    }
}

var filterGrid =
{
    rows_counter_text: " # event(s) found!"
}
function applyFilter() {
    var strImportanceFilter = "";
    strImportanceFilter += $('input[id="high"]')[0].checked ? "High;" : "";
    strImportanceFilter += $('input[id="medium"]')[0].checked ? "Medium;" : "";
    strImportanceFilter += $('input[id="low"]')[0].checked ? "Low" : "";
    var strCurrencyFilter = "";
    strCurrencyFilter += $('input[id="eur"]')[0].checked ? "EUR;" : "";
    strCurrencyFilter += $('input[id="usd"]')[0].checked ? "USD;" : "";
    strCurrencyFilter += $('input[id="jpy"]')[0].checked ? "JPY;" : "";
    strCurrencyFilter += $('input[id="gbp"]')[0].checked ? "GBP;" : "";
    strCurrencyFilter += $('input[id="chf"]')[0].checked ? "CHF;" : "";
    strCurrencyFilter += $('input[id="aud"]')[0].checked ? "AUD;" : "";
    strCurrencyFilter += $('input[id="cad"]')[0].checked ? "CAD;" : "";
    strCurrencyFilter += $('input[id="nzd"]')[0].checked ? "NZD;" : "";
    strCurrencyFilter += $('input[id="cny"]')[0].checked ? "CNY;" : "";


    setFilterGrid('e-cal-table', 1, filterGrid);

    TF_SetFilterValue('e-cal-table', 9, strCurrencyFilter);
    TF_SetFilterValue('e-cal-table', 4, strImportanceFilter);
    showFilter();
    TF_Filter('e-cal-table');
}

function applyFilterOnDateChanged() {
    if (TF_HasGrid('e-cal-table')) {
        var strImportanceFilter = "";
        strImportanceFilter += $('input[id="high"]')[0].checked ? "High;" : "";
        strImportanceFilter += $('input[id="medium"]')[0].checked ? "Medium;" : "";
        strImportanceFilter += $('input[id="low"]')[0].checked ? "Low" : "";
        var strCurrencyFilter = "";
        strCurrencyFilter += $('input[id="eur"]')[0].checked ? "EUR;" : "";
        strCurrencyFilter += $('input[id="usd"]')[0].checked ? "USD;" : "";
        strCurrencyFilter += $('input[id="jpy"]')[0].checked ? "JPY;" : "";
        strCurrencyFilter += $('input[id="gbp"]')[0].checked ? "GBP;" : "";
        strCurrencyFilter += $('input[id="chf"]')[0].checked ? "CHF;" : "";
        strCurrencyFilter += $('input[id="aud"]')[0].checked ? "AUD;" : "";
        strCurrencyFilter += $('input[id="cad"]')[0].checked ? "CAD;" : "";
        strCurrencyFilter += $('input[id="nzd"]')[0].checked ? "NZD;" : "";
        strCurrencyFilter += $('input[id="cny"]')[0].checked ? "CNY;" : "";

        TF_ClearFilters('e-cal-table');
        TF_RemoveFilterGrid('e-cal-table');

        setFilterGrid('e-cal-table', 1, filterGrid);
        TF_SetFilterValue('e-cal-table', 9, strCurrencyFilter);
        TF_SetFilterValue('e-cal-table', 4, strImportanceFilter);
        TF_Filter('e-cal-table');
    }
}

function clearFilter() {
    $('input[id="high"]')[0].checked = false;
    $('input[id="medium"]')[0].checked = false;
    $('input[id="low"]')[0].checked = false;
    $('input[id="eur"]')[0].checked = false;
    $('input[id="usd"]')[0].checked = false;
    $('input[id="jpy"]')[0].checked = false;
    $('input[id="gbp"]')[0].checked = false;
    $('input[id="chf"]')[0].checked = false;
    $('input[id="aud"]')[0].checked = false;
    $('input[id="cad"]')[0].checked = false;
    $('input[id="nzd"]')[0].checked = false;
    $('input[id="cny"]')[0].checked = false;
    if (TF_HasGrid('e-cal-table')) {
        TF_ClearFilters('e-cal-table');
        TF_RemoveFilterGrid('e-cal-table');
    }
    showFilter();

}

function SelectAllImportances(obj) {
    if (obj.checked) {
        //$('label[id="lblAllImportance"]')[0].textContent="Unselect All";
        document.getElementById('lblAllImportance').innerText = "Unselect All";
    } else {
        //$('label[id="lblAllImportance"]')[0].textContent="Select All";
        document.getElementById('lblAllImportance').innerText = "Select All";
    }
    $('input[id="high"]')[0].checked = obj.checked;
    $('input[id="medium"]')[0].checked = obj.checked;
    $('input[id="low"]')[0].checked = obj.checked;
}
function SelectAllCurrency(obj) {
    if (obj.checked) {
        //$('label[id="lblAllCurrency"]')[0].innerText("Unselect All");
        document.getElementById('lblAllCurrency').innerText = "Unselect All";
    } else {
        //$('label[id="lblAllCurrency"]')[0].innerText("Select All");
        document.getElementById('lblAllCurrency').innerText = "Select All";
    }
    $('input[id="eur"]')[0].checked = obj.checked;
    $('input[id="usd"]')[0].checked = obj.checked;
    $('input[id="jpy"]')[0].checked = obj.checked;
    $('input[id="gbp"]')[0].checked = obj.checked;
    $('input[id="chf"]')[0].checked = obj.checked;
    $('input[id="aud"]')[0].checked = obj.checked;
    $('input[id="cad"]')[0].checked = obj.checked;
    $('input[id="nzd"]')[0].checked = obj.checked;
    $('input[id="cny"]')[0].checked = obj.checked;
}

function splitStoryInPieces(story) {
    var index = story.indexOf(",\"f11");
    var storyContent = story.substring(0, index);
    var history = story.substring(index, story.length - index);

    var obj = jQuery.parseJSON(story);

}

var lastLoadedChartID;
var openedStories = new Array();
var controlDivToHide;
function ChangeHistoryTab(controlID, storyId, provider, parentID) {
    lastLoadedChartID = storyId;

    var requestID = parentID != undefined ? parentID : storyId;
    storyReq = getReqObjStory(requestID, provider, true);
    if ($('td[id="' + controlID + '"]').length > 0) {
        if (controlID.indexOf('Chart') != - 1) {

            otherControl = controlID.replace('Chart', 'History');
            otherControl1 = controlID.replace('Chart', 'Details');
            bodyControl = controlID.replace('Chart', 'Body');
            chartDivControl = controlID.replace('Chart', 'chartDiv');
            historyDivControl = controlID.replace('Chart', 'historyDiv');
            if ($('td[id="' + controlID + '"]')[0].className == "inactiveTab") {
                $('td[id="' + controlID + '"]')[0].className = "activeTab";
                if ($('td[id="' + otherControl + '"]').length > 0) {
                    $('td[id="' + otherControl + '"]')[0].className = "inactiveTab";
                }

                if ($('td[id="' + otherControl1 + '"]').length > 0) {
                    $('td[id="' + otherControl1 + '"]')[0].className = "inactiveTab";
                }

                if ($('td[id="' + bodyControl + '"]').length > 0) {
                    $('td[id="' + bodyControl + '"]')[0].className = "activeBody";
                }
                if ($('div[id="' + chartDivControl + '"]').length > 0) {
                    $('div[id="' + chartDivControl + '"]')[0].style.display = "block";
                }
                if ($('div[id="' + historyDivControl + '"]').length > 0) {
                    $('div[id="' + historyDivControl + '"]')[0].style.display = "none";
                }

            }


        } else if (controlID.indexOf('History') != - 1) {
            otherControl = controlID.replace('History', 'Chart');
            otherControl1 = controlID.replace('History', 'Details');
            bodyControl = controlID.replace('History', 'Body');
            chartDivControl = controlID.replace('History', 'chartDiv');
            historyDivControl = controlID.replace('History', 'historyDiv');
            if ($('td[id="' + controlID + '"]')[0].className == "inactiveTab") {
                $('td[id="' + controlID + '"]')[0].className = "activeTab";
                if ($('td[id="' + otherControl + '"]').length > 0) {
                    $('td[id="' + otherControl + '"]')[0].className = "inactiveTab";
                }
                if ($('td[id="' + otherControl1 + '"]').length > 0) {
                    $('td[id="' + otherControl1 + '"]')[0].className = "inactiveTab";
                }
                if ($('td[id="' + bodyControl + '"]').length > 0) {
                    $('td[id="' + bodyControl + '"]')[0].className = "activeBody";
                }
                if ($('div[id="' + chartDivControl + '"]').length > 0) {
                    if ((controlDivToHide != undefined && controlDivToHide.id == chartDivControl) || DoesStoryContainsID(openedStories, requestID)) {
                        $('div[id="' + chartDivControl + '"]')[0].style.display = "none";
                    }
                    controlDivToHide = $('div[id="' + chartDivControl + '"]')[0];
                }
                if ($('div[id="' + historyDivControl + '"]').length > 0) {
                    $('div[id="' + historyDivControl + '"]')[0].style.display = "block";
                }

            }
        }

        if (!DoesStoryContainsID(openedStories, requestID)) {
            cbfunctions.push({
                update: getStory,
                cmp: storyReq.i
            });
            openedStories.push(requestID);
        }

        connection.AddRequest(storyReq);
        connection.Flush();
    }
}

var theme = 'white';
function LoadCSSFiles() {
    var $ = document; // shortcut 
    theme = queryString("theme").toLowerCase();
    if (theme.toLowerCase() != 'black') {
        theme = 'white';
    }

    var cssId = 'myCss'; // you could encode the css path itself to generate id.. 
    if (!$.getElementById(cssId)) {
        var head = $.getElementsByTagName('head')[0];
        var link = $.createElement('link');
        link.id = cssId;
        link.rel = 'stylesheet';
        link.type = 'text/css';
        link.href = location.href.substring(0, location.href.lastIndexOf('/')) + '/css/' + theme + '/netd_p.css';
        link.media = 'all';
        head.appendChild(link);
        var head1 = $.getElementsByTagName('head')[0];
        var link1 = $.createElement('link');
        link1.id = cssId;
        link1.rel = 'stylesheet';
        link1.type = 'text/css';
        link1.href = location.href.substring(0, location.href.lastIndexOf('/')) + '/css/' + theme + '/netd.css';
        link1.media = 'all';
        head1.appendChild(link1);
    }
}

function ToDateTime(seconds) {
    var t = new Date(1970, 0, 1);
    t.setSeconds(seconds);
    return t;
}

function FillHistory(historyValues) {
    var historyTable = '<table><tr><td class="DateReleaseHeaderCls"><span>Release Date</span></td><td class="ValuesHeaderCls"><span>Actual</span></td><td class="ValuesHeaderCls"><span>Forecast</span></td><td class="ValuesHeaderCls"><span>Previous</span></td></tr><tr><td colspan="4" style="border: 1px solid transparent;"><hr /></td></tr>|RowsDetails</table>';
    for (var i = 0; i < historyValues.length; i++) {
        var historyRow = ' <tr><td class="DateReleaseCls"><span>|ReleaseDate</span></td><td class="ValuesCls" style="color:|FontColor"><span>|Actual</span></td><td class="ValuesCls"><span>|Forecast</span></td><td class="ValuesCls"><span>|Previous</span></td>';
        var color = theme == 'black' ? 'white' : 'black';
        if (i > 0) {
            if (historyValues[i][1] > historyValues[i - 1][1]) {
                color = 'green';
            } else if (historyValues[i][1] < historyValues[i - 1][1]) {
                color = 'red';
            }
        }

        historyRow = historyRow.replace("|ReleaseDate", historyValues[i][0]);
        historyRow = historyRow.replace("|FontColor", color);
        historyRow = historyRow.replace("|Actual", historyValues[i][1]);
        historyRow = historyRow.replace("|Forecast", historyValues[i][2]);
        historyRow = historyRow.replace("|Previous", historyValues[i][3]);
        historyTable = historyTable.replace("|RowsDetails", "|RowsDetails" + historyRow);

    }

    historyTable = historyTable.replace("|RowsDetails", "");

    if ($('div[id="historyDiv' + lastLoadedChartID + '"]').length > 0) {
        $('div[id="historyDiv' + lastLoadedChartID + '"]')[0].innerHTML = historyTable;
    }
    //$('div[id="chartDiv' + lastLoadedChartID + '"]').scrollIntoView();

}


$(document).ready(function () {
    var mouse_is_inside = false;

    //    if($('div[id="filterMenu"]').length>0) {
    //    debugger;
    //        $('div[id="filterMenu"]').hover(function(){  
    //            mouse_is_inside=true;  
    //        }, function(){  
    //        mouse_is_inside=false;  
    //    });

    $('div[id="filterMenu"]').live('mouseenter', function () {
        mouse_is_inside = true;
    }).live('mouseleave', function () {
        mouse_is_inside = false;
    });
    $('a[class="btnFilter"]').live('mouseenter', function () {
        mouse_is_inside = true;
    }).live('mouseleave', function () {
        mouse_is_inside = false;
    });

    $("body").mouseup(function () {

        if (!mouse_is_inside) {
            if ($('div[id="filterMenu"]').length > 0) {
                $('div[id="filterMenu"]')[0].style.display = 'none';
            }
        }
    });
});

function DoesStoryContainsID(story, id) {
    for (var i = 0; i < story.length; i++) {
        if (story[i] === id) {
            return true;
        }
    }
    return false;

}
