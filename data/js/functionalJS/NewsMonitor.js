Netdania = Netdania || {};
Netdania.News = Netdania.News || {};
Netdania.News.Monitor = Netdania.News.Monitor || {};


Netdania.News.Monitor.News = (function () {

    var _instance;

    function run() {

        // Singleton
        // Private methods and variables
        var _connection = null;
        var _newsTableId = '-newsTb';
        var _requestIds = new Array();
        var _requestInfo = new Array();

        var resultsNo = 0;


        var _addEvents = function () {

            _connection.addListener(Netdania.JsApi.Events.ONHISTORICALHEADLINES, function (newsHistoryResponse) {
                try {
                    if (Netdania.Util.isInArray(newsHistoryResponse.id, _requestIds)) {

                        _requestIds = $.grep(_requestIds, function (value) {
                            return value != newsHistoryResponse.id;
                        });

                        var news = [];
                        var _request = _getRequestData(newsHistoryResponse.id);
                        if (newsHistoryResponse.data !== null) {
                            news = _formatNewsData(newsHistoryResponse);
                        }
                        _request.updateCallback({news: news, control: _request.controlID, requestID: _request.id, type: _request.type });
                    }
                }
                catch (err) {
                    Netdania.Util.LogError("Netdania.News.Monitor.News.Events.ONHISTORICALHEADLINES", err);
                }
            });

            _connection.addListener(Netdania.JsApi.Events.ONHEADLINEUPDATE, function (monitorNewsResponse) {
                try {
                    if (monitorNewsResponse.data !== null) {
                        var _data = [];
                        _data.push(monitorNewsResponse.data);
                        monitorNewsResponse.data = _data;

                        var news = _formatNewsData(monitorNewsResponse);
                        var _request = _getRequestData(monitorNewsResponse.id);
                        _request.updateCallback({news: news, control: _request.controlID, requestID: _request.id, type: Netdania.News.Monitor.News.Type.UPDATE});

                    }
                }
                catch (err) {
                    Netdania.Util.LogError("Netdania.News.Monitor.News.Events.ONHEADLINEUPDATE", err);
                }
            });

            _connection.addListener(Netdania.JsApi.Events.ONNEWSSEARCH, function (newsHistoryResponse) {
                try {
                    if (Netdania.Util.isInArray(newsHistoryResponse.id, _requestIds)) {
                        _requestIds = $.grep(_requestIds, function (value) {
                            return value != newsHistoryResponse.id;
                        });

                        var news = [];
                        var _request = _getRequestData(newsHistoryResponse.id);
                        if (newsHistoryResponse.data !== null) {
                            news = _formatNewsData(newsHistoryResponse);
                        }
                        _request.updateCallback({news: news, control: _request.controlID, requestID: _request.id, type: _request.type});
                    }
                }
                catch (err) {
                    Netdania.Util.LogError("Netdania.News.Monitor.News.Events.ONNEWSSEARCH", err);
                }

            });

            _instance.addListener(Netdania.News.Monitor.News.Events.ALERTS_SEARCH, function (alertSearchResponse) {
                try {
                    var news = [];
                    var _request = _getRequestData(alertSearchResponse.id);
                    if (alertSearchResponse.data !== null) {
                        news = alertSearchResponse.data;
                    }
                    _request.updateCallback({news: news, control: _request.controlID, requestID: _request.id, type: _request.type});
                }
                catch (err) {
                    Netdania.Util.LogError("Netdania.News.Monitor.News.Events.ALERTS_SEARCH", err);
                }
            });

            _instance.addListener(Netdania.News.Monitor.News.Events.ALERTS_UPDATE, function (alertSearchResponse) {
                try {
                    var news = [];
                    var _request = _getRequestData(alertSearchResponse.id);
                    if (alertSearchResponse.data !== null) {
                        news = alertSearchResponse.data;
                    }
                    _request.updateCallback({news: news, control: _request.controlID, requestID: _request.id, type: Netdania.News.Monitor.News.Type.UPDATE});
                }
                catch (err) {
                    Netdania.Util.LogError("Netdania.News.Monitor.News.Events.ALERTS_UPDATE", err);
                }
            });
        };

        var _getRequestData = function (id) {
            for (var i = 0; i < _requestInfo.length; i++)
                if (_requestInfo[i].id == id) {
                    return _requestInfo[i];
                }
            return {};
        }

        var _formatNewsData = function (response) {
            var data = response.data;
            var requestId = response.id;
            var news = [];
            var _headlineObject, _r;
            var _eventObj = null;
            for (var lj = data.length - 1; lj >= 0; lj--) {
                _eventObj = null;
                _headlineObject = data[lj];
                var _request = _getRequestData(requestId);

                try {
                    _eventObj = JSON.parse(_headlineObject.h);
                } catch (ex) {

                } finally {
                }

                var alias = _request.alias;
                if (_request.alias == undefined) {
                    var relatedInstrumentsJSON;
                    var aliasData = {source: _request.source, provider: _request.provider};
                    alias = Netdania.News.Utilities.getAlias(aliasData);
                }

                if (_eventObj === null) {
                    _r = {
                        data: {
                            id: lj,
                            t: _headlineObject.t,
                            s: _headlineObject.s,
                            i: _headlineObject.i,
                            h: _headlineObject.h,
                            source: _request.source,
                            provider: _request.provider,
                            fullSource: _request.fullSource,
                            alias: alias
                        }
                    };
                }
                else {
                    var _headlineText = '';
                    if (_eventObj.f19 !== undefined && _eventObj.f13 !== undefined) {
                        var dateUnf = new Date(_eventObj.f15);
                        _headlineText = '<a href="' + _eventObj.f19 + '" target="_blank">' + Netdania.Util.dateFormat(dateUnf, "dd.mm.yyyy") + ' - ' + _eventObj.f13 + (_eventObj.f14 ? " - " + _eventObj.f14 : '') + (_eventObj.f20 ? '<br/>' + _eventObj.f20 : '') + '</a>';
                    }
                    else
                        _headlineText = _headlineObject.h.replace('"', '');
                    _r = {
                        data: {
                            id: lj,
                            t: _headlineObject.t,
                            s: _headlineObject.s,
                            i: _headlineObject.i,
                            h: _headlineText,
                            source: _request.source,
                            provider: _request.provider,
                            alias: alias
                        }
                    };
                }
                news.push(_r.data);
            }

            return news;

        }

        return {
            // Public methods and variables
            Init: function () {
                try {

                    Events.enable.call(this);

                    var host = Netdania.General.Config.Host;

                    var location = Netdania.Util.getLocation(Netdania.JsApi.Utilities.GetURL());
                    var connectionParams = Netdania.General.Config.GetConnectionParams(location.pathname, Netdania.General.Config.URLConnectionParams);
                    var connectionConfig = {
                        pollingInterval: connectionParams.pollingInterval, //milliseconds
                        host: host,
                        behavior: connectionParams.ct, //Netdania.JsApi.ConnectionType.LONGPOLLING, //LONG POLLING
                        type: 1, //the server response format: JSONP
                        v: 2
                    }

                    //create a json connection object
                    _connection = new Netdania.JsApi.JSONConnection(connectionConfig);
                    _connection.Connect();


                    _addEvents();
                }
                catch (err) {
                    Netdania.Util.LogError("Netdania.News.Monitor.News.Init", err);
                }
            },
            /*
             Check if control has requests and if It has re remove the requests
             */
            /*RemoveRequests: function (notificationImage, type) {
                var _removeRequest =[];
                $.each(_requestIds, function (i) {
                        var _request = _getRequestData(_requestIds[i]);
                        if (_request.controlID == notificationImage.id && _request.type == type ) {
                            _removeRequest.push(_requestIds[i]);
                        }
                    }
                );
                if (_removeRequest.length > 0) {
                    _connection.RemoveRequests(_removeRequest);
                    for (var i = 0; i < _removeRequest.length; i++) {
                        for (var j = 0; j < _requestIds.length; j++) {
                            if (_removeRequest[i] == _requestIds[j]) {
                                _requestIds.splice(j, 1);
                            }
                        }
                    }
                }
            },*/

            /*
             Check if control has requests and if It has re remove the requests
             */
            RemoveRequests: function (request) {
                try {
                    if (request.history != null && request.history != undefined) {
                        var monitorRemoveRequest = [];
                        $.each(request.history, function (i) {
                            monitorRemoveRequest.push(request.history[i]);
                        });
                        _connection.removeRequests(monitorRemoveRequest);
                        _connection.Flush();
                    }
                }
                catch (err) {
                    Netdania.Util.LogError("Netdania.News.Monitor.News.RemoveRequests", err);
                }
            },

            AddNewsSearchRequest: function (data) {
                try {
                    var notificationImage = data.control;
                    var start = data.start;
                    var end = data.end;
                    var headlines = data.headlines;
                    var _controlRequests = [];

                    var _needsLogin = false;
                    var _canSourceBeUsed = true;

                    for (var i = 0; i < notificationImage.source.length; i++) {
                        //check if the source needs the user be autheticated(login=true)
                        _needsLogin = notificationImage.source[i].login || false;
                        //for the sources that are used without login is possible to be removed when the user has logged in.
                        //ex. if FXWire realtime is used when the user is logged in and FXWire_dl delayed when the user is not logged in, then is no
                        //reason to use both realtime and delayed from fx wire when the user is logged in - so FXWire_dl will have useIfLoggedIn: false
                        var _useIfUserIsLoggedIn = notificationImage.source[i].useIfLoggedIn || false;
                        var _disabledByUser = notificationImage.source[i].userDisabled || false;
                        var _isAlert = notificationImage.source[i].alert || false;

                        _canSourceBeUsed = _needsLogin ? (notificationImage.user.authenticated ? true : false) : (notificationImage.user.authenticated ? _useIfUserIsLoggedIn : true);
                        //end check

                        /*
                         If we have a country filter parameter and country code exist
                         We append country code to the news source
                         */
                        var _countryFilter = notificationImage.source[i].countryFilter || false;
                        var _requestSource = notificationImage.source[i].source;
                        if (_countryFilter) {
                            var countryCode = notificationImage.user.countryCode;
                            if (countryCode != '') {
                                _requestSource = _requestSource + '^' + countryCode;
                            }
                        }

                        /*
                         If we have a country restriction parameter
                         We check if the user country match restriction and we remove the source otherwise
                         */
                        var _countryRestriction = notificationImage.source[i].countryRestriction || false;
                        if (_countryRestriction) {
                            var countryCode = notificationImage.user.countryCode;
                            if (_canSourceBeUsed && (countryCode == '' || countryCode.toLowerCase() != _countryRestriction.toLowerCase())) {
                                _canSourceBeUsed = false;
                            }
                        }

                        if (_canSourceBeUsed) {
                            if (!_disabledByUser) {
                                if (_isAlert) {
                                    var alertRequest = Netdania.Alert.SearchSentAlerts({dateStart: start, dateEnd: end, userID: notificationImage.user.id, control: notificationImage.control, headlines: headlines});
                                    _controlRequests.push(alertRequest.i);
                                    _requestInfo.push({ id: alertRequest.i, controlID: notificationImage.id, updateCallback: notificationImage.updateCallback, source: notificationImage.source[i].source, provider: notificationImage.source[i].provider, fullSource: _requestSource, alias: notificationImage.source[i].alias, type: Netdania.News.Monitor.News.Type.SEARCH });
                                    notificationImage.source[i].requested = true;
                                }
                                else {
                                    var newsRequest = Netdania.JsApi.Request.getReqObjNewsSearch(_requestSource + (notificationImage.isin ? '#' + notificationImage.isin : ''), '', headlines, start, end, 0, notificationImage.source[i].provider);
                                    _connection.AddRequest(newsRequest);
                                    _requestIds.push(newsRequest.i);
                                    _controlRequests.push(newsRequest.i);
                                    _requestInfo.push({
                                        id: newsRequest.i,
                                        controlID: notificationImage.id,
                                        updateCallback: notificationImage.updateCallback,
                                        source: notificationImage.source[i].source,
                                        provider: notificationImage.source[i].provider,
                                        fullSource: _requestSource,
                                        alias: notificationImage.source[i].alias,
                                        type: Netdania.News.Monitor.News.Type.SEARCH });
                                    notificationImage.source[i].requested = true;
                                }
                            }
                            else {
                                notificationImage.source[i].requested = false;
                            }

                            $.extend(notificationImage.source[i], {id: i, fullSource: _requestSource, isValid: true });
                        }
                    }
                    _connection.Flush();
                    return _controlRequests;
                }
                catch (err) {
                    Netdania.Util.LogError("Netdania.News.Monitor.News.AddNewsSearchRequest", err);
                }
            },

            AddNewsMonitorRequest: function (data) {
                try {

                    var notificationImage = data.control;
                    var headlines = data.headlines;

                    var _needsLogin = false;
                    var _canSourceBeUsed = true;
                    var _controlRequests = [];

                    for (var i = 0; i < notificationImage.source.length; i++) {
                        //check if the source needs the user be autheticated(login=true)
                        _needsLogin = notificationImage.source[i].login || false;
                        //for the sources that are used without login is possible to be removed when the user has logged in.
                        //ex. if FXWire realtime is used when the user is logged in and FXWire_dl delayed when the user is not logged in, then is no
                        //reason to use both realtime and delayed from fx wire when the user is logged in - so FXWire_dl will have useIfLoggedIn: false
                        var _useIfUserIsLoggedIn = notificationImage.source[i].useIfLoggedIn || false;
                        var _disabledByUser = notificationImage.source[i].userDisabled || false;
                        var _isAlert = notificationImage.source[i].alert || false;

                        _canSourceBeUsed = _needsLogin ? (notificationImage.user.authenticated ? true : false) : (notificationImage.user.authenticated ? _useIfUserIsLoggedIn : true);
                        //end check

                        /*
                         If we have a country filter parameter and country code exist
                         We append country code to the news source
                         */
                        var _countryFilter = notificationImage.source[i].countryFilter || false;
                        var _requestSource = notificationImage.source[i].source;
                        if (_countryFilter) {
                            var countryCode = notificationImage.user.countryCode;
                            if (countryCode != '') {
                                _requestSource = _requestSource + '^' + countryCode;
                            }
                        }

                        /*
                         If we have a country restriction parameter
                         We check if the user country match restriction and we remove the source otherwise
                         */
                        var _countryRestriction = notificationImage.source[i].countryRestriction || false;
                        if (_countryRestriction) {
                            var countryCode = notificationImage.user.countryCode;
                            if (_canSourceBeUsed && (countryCode == '' || countryCode.toLowerCase() != _countryRestriction.toLowerCase())) {
                                _canSourceBeUsed = false;
                            }
                        }

                        if (_canSourceBeUsed) {
                            if (!_disabledByUser) {
                                if (_isAlert) {
                                    var alertRequest = Netdania.Alert.HeadlinesSentAlerts({userID:notificationImage.user.id, control:notificationImage.control, headlines:headlines, maxNewsDisplayDays:notificationImage.maxNewsDisplayDays});
                                    _controlRequests.push(alertRequest.i);
                                    _requestInfo.push({ id: alertRequest.i, controlID: notificationImage.id, updateCallback: notificationImage.updateCallback, source: notificationImage.source[i].source, provider: notificationImage.source[i].provider, fullSource: _requestSource, alias: notificationImage.source[i].alias, type: Netdania.News.Monitor.News.Type.HEADLINE });
                                    notificationImage.source[i].requested = true;
                                }
                                else {
                                    var newsRequest = Netdania.JsApi.Request.getReqObjHeadlines(_requestSource + (notificationImage.isin ? '#' + notificationImage.isin : ''), headlines, notificationImage.source[i].provider, true);
                                    _connection.AddRequest(newsRequest);
                                    _requestIds.push(newsRequest.i);
                                    _controlRequests.push(newsRequest.i);
                                    _requestInfo.push({ id: newsRequest.i, controlID: notificationImage.id, updateCallback: notificationImage.updateCallback, source: notificationImage.source[i].source, provider: notificationImage.source[i].provider, fullSource: _requestSource, alias: notificationImage.source[i].alias, type: Netdania.News.Monitor.News.Type.HEADLINE });
                                    notificationImage.source[i].requested = true;
                                }
                            }
                            else {
                                notificationImage.source[i].requested = false;
                            }
                            $.extend(notificationImage.source[i], {id: i, fullSource: _requestSource, isValid: true });
                        }
                    }
                    _connection.Flush();
                    return _controlRequests;
                }
                catch (err) {
                    Netdania.Util.LogError("Netdania.News.Monitor.News.AddNewsMonitorRequest", err);
                }
            },
            WriteNews: function (notificationImage) {
                try {

                    if (notificationImage.news !== undefined && notificationImage.news.data !== undefined && notificationImage.news.data.length > 0) {
                        var headerClass, contentClass, width, fields, instr;

                        var fields = {
                            h: { name: 'Headline', value: '', width: 618, TDCls: 'newsMonitorTDCls', cellCls: 'newsMonitorHeadlineColumn' },
                            t: { name: 'Time', value: '', width: 100, TDCls: 'newsMonitorTDCls', cellCls: 'newsMonitorTimeColumn'}
                        };

                        var renderers = {

                            h: { fn: function (rec, cellid, val, oldval, fxy) {
                                var t = rec.t;
                                var dateUnf = new Date(t * 1000);
                                t = '';
                                var storyCls = (rec.s == 1) ? 'class ="nd-news-headline-row"' : 'class ="nd-news-headline-row nd-news-headline-row-nohover"';
                                if (rec.s != 1 && val.substring(0, 1) != "*") {
                                    val = "* " + val;
                                }
                                return t + '<span ' + storyCls + '>' + val.replace('RESEARCH: ', '') + '</span>';
                            }
                            },
                            t: { fn: function (rec, cellid, val, oldval, fxy) {
                                if (val === '' || val === "N/A" || val === undefined) {
                                    return "";
                                }
                                var dateUnf = new Date(val * 1000);

                                var relativeTimeString = Netdania.News.Utilities.timeDifference(new Date(), dateUnf);

                                return '<span>' + relativeTimeString + '</span>';
                            }
                            }
                        };

                        var _newsTableId = '-newsTb';

                        var instr = [];
                        instr = notificationImage.news.data;
                        var tableWidth = 0;
                        var headerHtml = '';
                        var lastField = '';
                        var groupByDate = false;
                        var contentClass = "newsMonitorDefaultContentClass";

                        for (var prop in fields) {
                            lastField = prop;
                        }
                        var cols = 0;
                        var align = "center";

                        var jshtml = '<table id="notification_table_' + notificationImage.id + '" class="nd-news-notification-table" border="0" cellspacing="0" cellpadding="0" >';
                        jshtml += '<tbody>';

                        var noDataMessage = "";

                        if (instr.length == 0) {
                            jshtml += '<tr><td colspan=' + cols + ' class="netd-grid-empty">' + noDataMessage + '</td></tr>';
                        }

                        var displayedNews = 0;

                        for (var ii = 0; ii < instr.length; ii++) {

                            if (displayedNews < notificationImage.maxDisplayEvents) {


                                /*
                                 Check for disabled or invalid sources
                                 */
                                var visible = true;

                                $.each(notificationImage.source, function (s) {
                                    if ((notificationImage.source[s].userDisabled || (notificationImage.source[s].isValid == undefined || notificationImage.source[s].isValid == false))  && notificationImage.source[s].source == instr[ii].source && notificationImage.source[s].provider == instr[ii].provider) {
                                        visible = false;
                                        return false;
                                    }
                                });

                                /*
                                 Check is the source is an alert
                                 */
                                var isAlert = false;
                                $.each(notificationImage.source, function (s) {
                                    if (notificationImage.source[s].source == instr[ii].source && notificationImage.source[s].provider == instr[ii].provider && notificationImage.source[s].alert) {
                                        isAlert = true;
                                        return false;
                                    }
                                });



                                if (instr[ii].disabled)
                                {
                                    visible = false;
                                }

                                if (visible) {

                                    displayedNews++;

                                    var recid = instr[ii].id;

                                    var trClass = contentClass;
                                    if (!instr[ii].read)
                                        trClass += " unread";

                                    var style = '';


                                    if (instr[ii].s == 1) {
                                        trClass += " clickable";
                                    }
                                    else{
                                        trClass += " flash";
                                    }

                                    instr[ii]["idrow"] = notificationImage.id - instr[ii].id;

                                    if (isAlert){
                                        jshtml += '<tr onclick="Netdania.News.Monitor.News.onStoryAlertClick(this, event)" p="' + instr[ii].provider + '" s="' + instr[ii].s + '" id="' + instr[ii]["idrow"] + '" i="' + instr[ii].i.replace(/"/g, '%22') + '" source="' + instr[ii].source + '" alias="' + instr[ii].alias + '"  t="' + instr[ii].t + '" class="' + trClass + '" ' + style + '>';
                                    }   else{
                                        jshtml += '<tr onclick="Netdania.News.Monitor.News.onStoryClick(this, event)" p="' + instr[ii].provider + '" s="' + instr[ii].s + '" id="' + instr[ii]["idrow"] + '" i="' + instr[ii].i.replace(/"/g, '%22') + '" source="' + instr[ii].source + '" alias="' + instr[ii].alias + '"  t="' + instr[ii].t + '"  class="' + trClass + '" ' + style + '>';
                                    }



                                    var cellid = recid + '-h';
                                    var value = instr[ii].h;
                                    if (value === undefined) {
                                        value = '';
                                    }
                                    if (renderers['h'] !== undefined) {
                                        value = renderers['h'].fn(instr[ii], cellid, value, value, 'h');
                                    } else {
                                        value = '<div>' + value + '</div>';
                                    }
                                    var lc = '';
                                    if (lastField === 'h') {
                                        lc = ' lastColCls';
                                    }
                                    jshtml += '<td style="overflow: auto;" class="' + fields['h'].TDCls + lc + '"><div id="' + cellid + '" class="' + fields['h'].cellCls + '" >' + value + '</div>';


                                    var alias = (instr[ii].alias == undefined ? '' : instr[ii].alias);
                                    jshtml += '<div class="newsMonitorAliasColumn"><span>By ' + alias + '</span></div>';


                                    var cellid = recid + '-t';
                                    var value = instr[ii].t;
                                    if (value === undefined) {
                                        value = '';
                                    }
                                    if (renderers['t'] !== undefined) {
                                        value = renderers['t'].fn(instr[ii], cellid, value, value, 't');
                                    } else {
                                        value = '<div>' + value + '</div>';
                                    }
                                    var lc = '';
                                    if (lastField === 't') {
                                        lc = ' lastColCls';
                                    }
                                    jshtml += '<div class="' + fields['t'].cellCls + '"><span>' + value + '</span></div>';


                                    jshtml += '</td></tr>';

                                    instr[ii]["read"] = true;
                                }
                            }
                        }

                        if (displayedNews == 0){
                            jshtml += '<tr><td>No events found</td></tr>';
                        }

                        jshtml += '</tbody>';
                        jshtml += '</table>';


                        notificationImage.news.html = jshtml;

                    }
                    else {
                        notificationImage.news = notificationImage.news || {};
                        notificationImage.news.data = notificationImage.news.data || [];
                        notificationImage.news.html = "<div class='nd-notifications-nodata'>No events found</div>";
                    }
                } catch (err) {
                    Netdania.Util.LogError("Netdania.News.Monitor.News.WriteNews", err);
                }
            },
            WriteNewsList: function (notificationImage) {

                var readNews = 0;
                var displayedNews = 0;
                try {

                    if (notificationImage.news !== undefined && notificationImage.news.data !== undefined && notificationImage.news.data.length > 0) {
                        var headerClass, contentClass, width, fields, instr;

                        var fields = {
                            h: { name: 'Headline', value: '', width: 618, TDCls: 'newsMonitorTDCls', cellCls: 'newsMonitorHeadlineColumn' },
                            t: { name: 'Time', value: '', width: 100, TDCls: 'newsMonitorTDCls', cellCls: 'newsMonitorTimeColumn'}
                        };

                        var renderers = {

                            h: { fn: function (rec, cellid, val, oldval, fxy) {
                                var t = rec.t;
                                var dateUnf = new Date(t * 1000);
                                t = '';
                                var storyCls = (rec.s == 1) ? 'class ="nd-news-headline-row"' : 'class ="nd-news-headline-row nd-news-headline-row-nohover"';
                                if (rec.s != 1 && val.substring(0, 1) != "*") {
                                    val = "* " + val;
                                }
                                return t + '<span ' + storyCls + '>' + val.replace('RESEARCH: ', '') + '</span>';
                            }
                            },
                            t: { fn: function (rec, cellid, val, oldval, fxy) {
                                if (val === '' || val === "N/A" || val === undefined) {
                                    return "";
                                }
                                var dateUnf = new Date(val * 1000);

                                var relativeTimeString = Netdania.Util.dateFormat(dateUnf, "dd mmm. yyyy HH:MM:ss").toLowerCase();

                                return '<span>' + relativeTimeString + '</span>';
                            }
                            }
                        };

                        var _newsTableId = '-newsTb';

                        var instr = [];
                        instr = notificationImage.news.data;
                        var tableWidth = 0;
                        var headerHtml = '';
                        var lastField = '';
                        var groupByDate = false;
                        var contentClass = "newsMonitorDefaultContentClass";

                        for (var prop in fields) {
                            lastField = prop;
                        }
                        var cols = 0;
                        var align = "center";

                        var jshtml = '<table id="notification_table_' + notificationImage.id + '" class="nd-news-notification-table" border="0" cellspacing="0" cellpadding="0" >';
                        jshtml += '<tbody>';

                        var noDataMessage = "";

                        if (instr.length == 0) {
                            jshtml += '<tr><td colspan=' + cols + ' class="netd-grid-empty">' + noDataMessage + '</td></tr>';
                        }


                        notificationImage.list.limit = false;

                        for (var ii = 0; ii < instr.length; ii++) {

                            if (displayedNews < notificationImage.list.maxDisplayEvents) {

                                /*
                                 Check for disabled or invalid sources
                                 */
                                var visible = true;

                                $.each(notificationImage.source, function (s) {
                                    if ((notificationImage.source[s].userDisabled || (notificationImage.source[s].isValid == undefined || notificationImage.source[s].isValid == false))  && notificationImage.source[s].source == instr[ii].source && notificationImage.source[s].provider == instr[ii].provider) {
                                        visible = false;
                                        return false;
                                    }
                                });


                                /*
                                 Check is the source is an alert
                                 */
                                var isAlert = false;
                                $.each(notificationImage.source, function (s) {
                                    if (notificationImage.source[s].source == instr[ii].source && notificationImage.source[s].provider == instr[ii].provider && notificationImage.source[s].alert) {
                                        isAlert = true;
                                        return false;
                                    }
                                });


                                if (instr[ii].disabled)
                                {
                                    visible = false;
                                }

                                if (visible) {

                                    displayedNews++;

                                    var recid = instr[ii].id;

                                    var trClass = contentClass;
                                    if (!instr[ii].read)
                                        trClass += " unread";

                                    if (instr[ii].s == 1) {
                                        trClass += " clickable";
                                    }
                                    else{
                                        trClass += " flash";
                                    }

                                    var style = '';

                                    instr[ii]["idrow"] = notificationImage.id + '-row-' + instr[ii].id;

                                    if (isAlert){
                                        jshtml += '<tr onclick="Netdania.News.Monitor.News.onStoryAlertClick(this, event)" p="' + instr[ii].provider + '" s="' + instr[ii].s + '" id="' + instr[ii]["idrow"] + '" i="' + instr[ii].i.replace(/"/g, '%22') + '" source="' + instr[ii].source + '" alias="' + instr[ii].alias + '"  t="' + instr[ii].t + '" class="' + trClass + '" ' + style + '>';
                                    }   else{
                                        jshtml += '<tr onclick="Netdania.News.Monitor.News.onStoryClick(this, event)" p="' + instr[ii].provider + '" s="' + instr[ii].s + '" i="' + instr[ii].i.replace(/"/g, '%22') + '" id="' + instr[ii]["idrow"] + '" source="' + instr[ii].source + '" alias="' + instr[ii].alias + '"  t="' + instr[ii].t + '"  class="' + trClass + '" ' + style + '>';
                                    }



                                    var cellid = recid + '-h';
                                    var value = instr[ii].h;
                                    if (value === undefined) {
                                        value = '';
                                    }
                                    if (renderers['h'] !== undefined) {
                                        value = renderers['h'].fn(instr[ii], cellid, value, value, 'h');
                                    } else {
                                        value = '<div>' + value + '</div>';
                                    }
                                    var lc = '';
                                    if (lastField === 'h') {
                                        lc = ' lastColCls';
                                    }
                                    jshtml += '<td style="overflow: auto;" class="' + fields['h'].TDCls + lc + '"><div id="' + cellid + '" class="' + fields['h'].cellCls + '" >' + value + '</div>';


                                    var alias = (instr[ii].alias == undefined ? '' : instr[ii].alias);
                                    jshtml += '<div class="newsMonitorAliasColumn"><span>By ' + alias + '</span></div>';


                                    var cellid = recid + '-t';
                                    var value = instr[ii].t;
                                    if (value === undefined) {
                                        value = '';
                                    }
                                    if (renderers['t'] !== undefined) {
                                        value = renderers['t'].fn(instr[ii], cellid, value, value, 't');
                                    } else {
                                        value = '<div>' + value + '</div>';
                                    }
                                    var lc = '';
                                    if (lastField === 't') {
                                        lc = ' lastColCls';
                                    }
                                    jshtml += '<div class="' + fields['t'].cellCls + '"><span>' + value + '</span></div>';


                                    jshtml += '</td></tr>';

                                    notificationImage.list.lastNewsTime = instr[ii].t;

                                    if (!instr[ii]["read"]){
                                        instr[ii]["read"] = true;
                                        readNews++;
                                    }

                                    /*If we display the last news (the limit has been reached*/
                                    if (notificationImage.news.limitID != null && notificationImage.news.limitID != undefined && notificationImage.news.limitID == instr[ii].i){
                                        notificationImage.list.limit = true;
                                    }

                                }


                            }
                        }

                        if (displayedNews == 0){
                            jshtml += '<tr><td>No events found</td></tr>';
                        }

                        jshtml += '</tbody>';
                        jshtml += '</table>';


                        notificationImage.list.htmlText = jshtml;


                    }
                    else {
                        notificationImage.list.htmlText = "<div class='nd-notifications-nodata'>No events found</div>";
                    }
                } catch (err) {
                    Netdania.Util.LogError("Netdania.News.Monitor.News.WriteNews", err);
                }

                notificationImage.list.readNews = readNews;
                notificationImage.list.displayedNews = displayedNews;
            }
        };
    };


    return {
        // Get the Singleton instance if one exists
        // or create one if it doesn't
        getInstance: function () {

            if (!_instance) {
                _instance = run();
                _instance.Init();
            }

            return _instance;
        },
        onStoryClick: function (news, event) {
            if ($(news).attr("s") == 1) {
                var config = new Netdania.News.General.Config();
                var stringUrl = config.NewsStory + '?id=' + $(news).attr("i") + '&p=' + $(news).attr("p") + '&s=' + $(news).attr("source") + '&t=' + $(news).attr("t") + '&h=' + encodeURIComponent($($(news).find('.nd-news-headline-row')[0]).html());
                window.location.href = stringUrl;
            }

        },
        onStoryAlertClick: function (news, event) {
            var stringUrl = Netdania.Alert.Config.AlertDetail + '?id=' + $(news).attr("i");
            window.location.href = stringUrl;

        },

        Events: {
            NEWSUPDATE: 'OnMonitorNewsUpdate',
            ALERTS_UPDATE: 'OnMonitorAlertsUpdate',
            ALERTS_SEARCH: 'OnMonitorAlertsSearch',
            ALERTS_UPDATE: 'OnMonitorAlertsUpdate'
        },
        Type: {
            SEARCH: 'Search',
            UPDATE: 'Update',
            HEADLINE: 'Headline'
        }




    };
})();

Netdania.News.Monitor.Storage = (function () {
    var _storageVersion = '1.4'
    var _storageName = 'monitoredNews';
    var _anonymousUserID = 'anonymous';
    var _newsObject = {};
    var maxDataStorageDays = 30;
    var runCleanEveryDays = 1;
    /*
     Return the object from storage
     */
    var GetNewsFromStorage = function () {
        var defaultObject = { users: [], version:_storageVersion};

        var storageObject = defaultObject;

        try {
            var alertsString = $.ezstorage.get(_storageName);
            if (alertsString !== undefined) {
                if (typeof alertsString == "string") {
                    storageObject = JSON.parse(alertsString);
                    if (storageObject !== undefined && typeof storageObject == "string") {
                        storageObject = JSON.parse(storageObject);
                    }
                }
                else {
                    storageObject = alertsString;
                }

                /*
                Check if we have the same version
                 */
                if (storageObject == null || storageObject == undefined || storageObject.version == null || storageObject.version == undefined || storageObject.version != _storageVersion){
                        storageObject =  defaultObject;
                }

            }
        } catch (err) {
            Netdania.Util.LogError("News.Monitor.Storage.GetNewsFromStorage", err);
        }
        return  storageObject;
    }

    /*
     Write the data to local storage
     */
    var SetNewsToStorage = function () {
        try {
            $.ezstorage.set(_storageName, _newsObject, {persist: true});
        } catch
            (err) {
            Netdania.Util.LogError("News.Monitor.Storage.SetNewsToStorage", err);
        }
    }

    /*
     Delete the data from storage
     */
    var DeleteNewsFromStorage = function () {
        try {
            $.ezstorage.remove(_storageName);
        } catch
            (err) {
            Netdania.Util.LogError("News.Monitor.Storage.DeleteNewsFromStorage", err);
        }
    }

    /*
     Takes oll the storage data and check the item time stamp to be expired
     Old controls data that are no more used will be deleted
     */
    var CleanNewsStorage = function () {
        try {
            var currentTime = new Date().getTime() / 1000;

            _newsObject = GetNewsFromStorage();

            $.each(_newsObject.users, function (u) {
                var userControls = _newsObject.users[u].controls;
                $.each(userControls, function (i) {
                    if (userControls[i] != null && userControls[i] != undefined) {
                        if (userControls[i].updateTime != null && userControls[i].updateTime != undefined) {
                            var storageTime = userControls[i].updateTime;
                            var maxTime = storageTime + (maxDataStorageDays * 24 * 60 * 60)
                            if (maxTime < currentTime) {
                                userControls.splice(i, 1);
                                SetNewsToStorage();
                            }
                        }
                        else {
                            userControls.splice(i, 1);
                            SetNewsToStorage();
                        }
                    }
                });
            });
        } catch
            (err) {
            Netdania.Util.LogError("News.Monitor.Storage.CleanNewsStorage error", err);
        }
    }

    /*
     Take an image object and extract only the id of the control and the id's of the read news
     This will be the object stored
     */
    var _ImageToStorageDate = function (image) {
        try {
            var updateTime = new Date().getTime() / 1000;
            /*
             Store disabled sources.
             On page load we will compare the control source with this sources
             */
            var sourceArray = [];
            if (image.source != undefined) {
                $.each(image.source, function (n) {
                    if (image.source[n].userDisabled) {
                        sourceArray.push({source: image.source[n].source, provider: image.source[n].provider, userDisabled: image.source[n].userDisabled});
                    }
                });
            }
            return {
                id: image.id,
                news: {
                    data: image.news.data,
                    source: sourceArray,
                    lastNewsTime: image.news.lastNewsTime,
                    firstNewsTime: image.news.firstNewsTime
                },
                updateTime: updateTime};
        } catch
            (err) {
            Netdania.Util.LogError("News.Monitor.Storage.ImageToStorageDate", err);
        }
        return {};
    }

    var _persistentData = function (oldData, newData) {
        try {
            /*
            Keep de not changing data
             */
            if (oldData.news.firstNewsTime != null && oldData.news.firstNewsTime != undefined) {
                newData.news.firstNewsTime = oldData.news.firstNewsTime;
            }

            return newData;
        }
        catch
            (err) {
            Netdania.Util.LogError("News.Monitor.Storage.MergeData", err);
        }
    };

    var _userControls = function (users, userID) {

        var controls;
        try {
            if (userID == null || userID == undefined) {
                userID = _anonymousUserID;
            }

            var found = false;
            $.each(users, function (i) {
                /*Find user match*/
                if (userID == users[i].id) {
                    users[i].controls = users[i].controls || [];
                    controls = users[i].controls;
                    found = true;
                    return false;
                }
            });

            if (!found) {
                users.push({id: userID, controls: []});
                controls = users[0].controls;
            }
        }
        catch
            (err) {
            Netdania.Util.LogError("News.Monitor.Storage.UserControls", err);
        }

        return controls;
    };


    return {
        /*
         Return the object from the storage for a specific image id
         */
        GetNewsData: function (data) {

            var alert = undefined;
            try {

                /*Get object from storage*/
                _newsObject = GetNewsFromStorage();

                if (_newsObject.users != null && _newsObject.users != undefined && data.authenticated) {
                    /*Get controls for user*/
                    var userControls = _userControls(_newsObject.users, data.idUser);

                    $.each(userControls, function (i) {
                        /* Check the ID of the control */
                        if (userControls[i].id == data.idControl) {
                            alert = userControls[i].news;
                            return false;
                        }

                    });
                }
            } catch
                (err) {
                Netdania.Util.LogError("News.Monitor.Storage.GetNewsData", err);
            }
            return alert;
        },
        /*
         Update the date to the storage if the image id is found
         Or Insert the new data to storage
         */
        UpdateNews: function (Image) {
            try {
                /*Get object from storage*/
                _newsObject = GetNewsFromStorage();


                if (_newsObject.users != null && _newsObject.users != undefined && Image.user.authenticated) {
                    /*Get controls for user*/
                    var userControls = _userControls(_newsObject.users, Image.user.id);

                    var storageData = _ImageToStorageDate(Image);
                    var boolFound = false;
                    $.each(userControls, function (i) {
                        if (userControls[i].id == Image.id) {
                            userControls[i] = storageData;// _persistentData(userControls[i], storageData);
                            SetNewsToStorage();
                            boolFound = true;
                            return false;
                        }
                    });

                    if (!boolFound) {
                        userControls.push(storageData);
                        SetNewsToStorage();
                    }
                }

                /*
                 Run the clean
                 */
                CleanNewsStorage();
            } catch
                (err) {
                Netdania.Util.LogError("News.Monitor.Storage.UpdateNews", err);
            }
        }
    }

})();













