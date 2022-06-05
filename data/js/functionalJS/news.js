Netdania = Netdania || {};
Netdania.NewsComponent = Netdania.NewsComponent || {};

Netdania.NewsComponent.RenderNews = function (config) {
    var _source = config.source || 'GoogleForex';
    var _provider = config.provider || 'rssnews';
    var _theme = config.theme || '';
    var _isLocalDS = config.isLocalDS || false;
    var _localDS = config.localDS;
    var _self = this;
    var _connection = null;
    var _newsTableId = '-newsTb';
    var _requestIds = new Array();
    //contains source, provider, requestId
    var _requestInfo = new Array();
    var _removeNewsOnUpdate = config.removeNewsOnUpdate || false; //if true when an update is received the oldest news is removed
    var _storyRender = config.storyRender;
    var _self = this;
    var resultsNo = 0;
    var _itemsOnPage = config.pagination ? (config.itemsOnPage || 10) : (config.maxHeadlines || 10);
    var _showAlias = config.showAlias === undefined ? true : config.showAlias;

    _self.NEWSRECEIVEDEVENT = 'NewsReceivedEvent';
    _self.NONEWSEVENT = "NoNewsEvent";

    var pagination = { addNewInterval: true, pageNo: 1, intervals: [] };

    Events.enable.call(this);


    this.initNews = function () {

        var host = Netdania.General.Config.Host; //the NetDania server to connect to


        var location = Netdania.Util.getLocation(Netdania.JsApi.Utilities.GetURL());
        var connectionParams = Netdania.General.Config.GetConnectionParams(location.pathname, Netdania.General.Config.URLConnectionParams);
        var connectionConfig = {
            pollingInterval: connectionParams.pollingInterval, //milliseconds
            host: host,
            behavior: connectionParams.ct, //Netdania.JsApi.ConnectionType.LONGPOLLING, //LONG POLLING
            type: 1, //the server response format: JSONP
            v: 1
        }

        //create a json connection object
        _connection = new Netdania.JsApi.JSONConnection(connectionConfig);
        _connection.Connect();

        var cookies = Netdania.Util.Cookies();
        if (cookies.getCookie("backNewsStory") == "1" && cookies.getCookie("pagination")) {
            pagination = jQuery.parseJSON(cookies.getCookie("pagination"));
            pagination.addNewInterval = false;
            config.start = pagination.intervals[pagination.pageNo - 1];
            config.end = (pagination.intervals[pagination.pageNo - 2] === undefined) ? Math.round(new Date().getTime() / 1000) : pagination.intervals[pagination.pageNo - 2];
            cookies.setCookie("backNewsStory", "0", "");
        }

        cookies.setCookie("pagination", "", "");

        return _renderNews();
    }

    this.init = function () {

    }

    this.update = function () {

    }

    var _logResponse = function (divId, data, rt, id, type) {
        var div = document.getElementById(divId);
        var newdiv = document.createElement('div');
        newdiv.innerHTML += JSON.stringify(data) + "<br/>";
        newdiv.innerHTML += "<span style=\"color:red\">timestamp: " + (rt || "") + "</span><br/>";
        newdiv.innerHTML += "<span style=\"color:green\">request id: " + (id || "") + "</span><br/>";
        newdiv.innerHTML += "<span style=\"color:blue\">request type: " + (type || "") + "</span><br/>";
        div.appendChild(newdiv);
    }

    var _renderNewsTable = function (tableConfig, startFrame, endFrame) {
        var headerClass, contentClass, contentOddClass, width, fields, instr;

        var fields = {
            h: { name: 'Headline', value: '', width: 618, hdTDcls: 'hdTDClsLeft' + _theme, TDCls: 'newsTDClsLeft' + _theme, cellCls: 'newsHeadlineColumn' + _theme, hdCellCls: 'nd_fq_row' + _theme },
            t: { name: 'Time', value: '', width: 100, hdTDcls: 'hdTDClsLeft' + _theme, TDCls: 'newsTDClsLeft' + _theme, cellCls: 'newsTimeColumn' + _theme, hdCellCls: 'nd_fq_row' + _theme },
            source: { name: 'Source', value: '', width: 100, hdTDcls: 'hdTDClsLeft' + _theme, TDCls: 'newsTDClsLeft' + _theme, cellCls: 'newsHeadlineColumn' + _theme, hdCellCls: 'nd_fq_row' + _theme },
            alias: { name: 'Alias', value: '', width: 100, hdTDcls: 'hdTDClsLeft' + _theme, TDCls: 'newsTDClsLeft' + _theme, cellCls: 'newsHeadlineColumn' + _theme, hdCellCls: 'nd_fq_row' + _theme }
        };

        var renderers = {

            h: { fn: function (grid, rec, cellid, val, oldval, fxy) {
                var t = rec.data.t;
                var dateUnf = new Date(t * 1000);
                t = ''; // '<span class="nd-headline-time nd-left">' + Netdania.Util.dateFormat(dateUnf, "HH:MM:ss").toLowerCase() + '</span>';
                var storyCls = (rec.data.s == 1) ? 'class ="nd-news-headline-row"' : 'class ="nd-news-headline-row-nohover"';
                return t + '<span ' + storyCls + '>' + val.replace('RESEARCH: ', '') + '</span>';
            }
            },
            t: { fn: function (grid, rec, cellid, val, oldval, fxy) {
                if (val === '' || val === "N/A" || val === undefined) {
                    return "";
                }
                var dateUnf = new Date(val * 1000);

                return '<span>' + Netdania.Util.dateFormat(dateUnf, "dd.mm.yy").toLowerCase() + '</span>' +
                    '<span style="padding-left:5px">' + Netdania.Util.dateFormat(dateUnf, "HH:MM").toLowerCase() + '</span>';
            }
            }
        };

        var _newsTableId = '-newsTb';

        var instr = tableConfig.records;
        var tableWidth = 0;
        var headerHtml = '';
        var colsHtml = '';
        var lastField = '';
        var contentClass = tableConfig.contentClass;
        if (tableConfig.contentClass === undefined || tableConfig.contentClass === null || tableConfig.contentClass === '') {
            contentClass = "gridDefaultContentClass";
        }
        var contentOddClass = tableConfig.contentOddClass;
        if (tableConfig.contentOddClass === undefined || tableConfig.contentOddClass === null || tableConfig.contentOddClass === '') {
            contentOddClass = "gridDefaultContentOddClass";
        }

        if (tableConfig.width !== undefined && tableConfig.width !== null) {
            tableWidth = tableConfig.width;
        }
        for (var prop in fields) {
            lastField = prop;
        }
        var cols = 0;

        var forceHeight = '';

        var align = '';
        if (tableConfig.align === '') {
            align = "center";
        }
        else {
            align = tableConfig.align;
        }
        var jshtml = '';
        if (startFrame !== undefined) {
            jshtml = startFrame + '<table id="' + tableConfig.renderTo + _newsTableId + '" class="nd-news-table"   border="0" cellspacing="0" cellpadding="0" align="' + align + '">';
        }
        else {
            jshtml = '<table id="' + tableConfig.renderTo + _newsTableId + '" class="nd-news-table" border="0" cellspacing="0" cellpadding="0" >';
        }
        jshtml += colsHtml;
        var t = jshtml;
        if (tableConfig.hideHeaders !== true) {
            var headerCls = '';
            if (tableConfig.headerCls !== undefined) {
                headerCls = 'class="' + tableConfig.headerCls + '"';
            }
            jshtml += '<thead ' + headerCls + '><tr>';
            jshtml += headerHtml;
            jshtml += '</tr></thead>';
        }
        //jshtml += '</table>';

        //jshtml += '<div ' + forceHeight + ' class="news-table">';
        //jshtml += t;
        jshtml += '<tbody>';
        var gday = -1;
        var gmonth = -1;
        var gyear = -1;
        if (tableConfig.groupByDate === true && instr.length > 0) {
            gd = new Date(instr[0].data.t * 1000);
        }

        var noDataMessage = tableConfig.noDataMessage;
        if (noDataMessage === undefined) {
            noDataMessage = '';
        }
        if (instr.length == 0) {
            jshtml += '<tr><td colspan=' + cols + ' class="netd-grid-empty">' + noDataMessage + '</td></tr>';
        }
        for (var ii = 0; ii < instr.length; ii++) {
            if (tableConfig.groupByDate === true) {
                var cd = new Date(instr[ii].data.t * 1000);
                if (cd.getDay() !== gday || cd.getMonth() !== gmonth || cd.getYear() !== gyear) {
                    jshtml += '<tr><td colspan=' + cols + ' class="netd-group-hd ">' + Netdania.Util.dateFormat(cd, "dd. mmmm yyyy") + '</td></tr>';
                    gday = cd.getDay();
                    gmonth = cd.getMonth();
                    gyear = cd.getYear();
                }
            }

            var recid = instr[ii].id;

            // for (var prop1 in fields) {
            //     if (fields.hasOwnProperty(prop1)) {
            var relatedInstrumentsJSON;
            try {
                relatedInstrumentsJSON = jQuery.parseJSON(instr[ii].data.i.substring(instr[ii].data.i.indexOf("{"), instr[ii].data.i.length));
            }
            catch (exception) {

            }
            var alias = Netdania.News.Utilities.getAliasBySource(instr[ii].data.source, relatedInstrumentsJSON);

            if (!_showAlias) alias = '';

            jshtml += '<tr p="' + instr[ii].data.provider + '" s="' + instr[ii].data.s + '" id="' + instr[ii].data.i.replace(/"/g, '%22') + '" source="' + instr[ii].data.source + '" alias="' + alias + '"  t="' + instr[ii].data.t + '"  openFQ="' + instr[ii].data.openFQ + '" class="' + contentClass + '">';
            var cellid = recid + '-h';
            var value = instr[ii].data['h'];
            if (value === undefined) {
                value = '';
            }
            if (renderers['h'] !== undefined) {
                value = renderers['h'].fn(tableConfig, instr[ii], cellid, value, value, 'h');
            } else {
                value = '<div>' + value + '</div>';
            }
            var lc = '';
            if (lastField === 'h') {
                lc = ' lastColCls';
            }

            jshtml += '<td style="overflow: auto;" class="' + fields['h'].TDCls + lc + '"><img src = "/JSComponents/News/General/images/source.png"></td>';
            jshtml += '<td style="overflow: auto;" class="' + fields['h'].TDCls + lc + '"><div id="' + cellid + '" class="' + fields['h'].cellCls + '" >' + value + '</div></td>';
            jshtml += '</tr>';

            jshtml += '<tr class="' + contentOddClass + '"><td colspan="2">';

            var cellid = recid + '-source';
            var value = alias;
            if (value === undefined) {
                value = '';
            }
            if (renderers['source'] !== undefined) {
                value = renderers['source'].fn(tableConfig, instr[ii], cellid, value, value, 'source');
            } else {
                value = '<span>' + value + '</span>';
            }
            var lc = '';
            if (lastField === 'source') {
                lc = ' lastColCls';
            }
            if (relatedInstrumentsJSON !== undefined && relatedInstrumentsJSON.Inf && relatedInstrumentsJSON.Inf.RNA && relatedInstrumentsJSON.Inf.RNA != '') {
                value = 'by ' + Netdania.News.Utilities.getAliasBySource(instr[ii].data.source, relatedInstrumentsJSON) + ',<span class="nd-news-delivered-span">Delivered by ' + Netdania.News.Utilities.getAliasBySource(instr[ii].data.source, undefined) + '</span>';
            }
            else
                value = 'Delivered by ' + value;
            jshtml += '<span id="' + cellid + '" class="' + fields['h'].TDCls + '" >' + value + '  </span>';
            var cellid = recid + '-t';
            var value = instr[ii].data['t'];
            if (value === undefined) {
                value = '';
            }
            if (renderers['t'] !== undefined) {
                value = renderers['t'].fn(tableConfig, instr[ii], cellid, value, value, 't');
            } else {
                value = '<div>' + value + '</div>';
            }
            var lc = '';
            if (lastField === 't') {
                lc = ' lastColCls';
            }
            jshtml += '<span id="' + cellid + '" class="' + fields['h'].TDCls + '">' + value + '</span>';
            jshtml += '</td></tr>';

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
            el.innerHTML = jshtml;
            var _newsTable = $('#' + tableConfig.renderTo + _newsTableId);
            var newsConfig = new Netdania.News.General.Config();
            $('#' + tableConfig.renderTo + _newsTableId + ' .' + contentClass).click(function () {
                if ($(this).attr("openFQ") !== '' && $(this).attr("openFQ") !== 'undefined') {
                    id = decodeURIComponent($(this).attr("id"));
                    var relatedInstrumentsJSON;
                    try {
                        relatedInstrumentsJSON = jQuery.parseJSON(id.substring(id.indexOf("{"), id.length));
                    }
                    catch (exception) {

                    }
                    if (relatedInstrumentsJSON) {
                        if (relatedInstrumentsJSON.Syms.length == 1)
                            window.location.href = newsConfig.FullQuote + '?symbol=' + relatedInstrumentsJSON.Syms[0].sym + ((relatedInstrumentsJSON.Syms[0].prv !== undefined) ? '&provider=' + relatedInstrumentsJSON.Syms[0].prv : '');
                        return;
                    }
                }

                if ($(this).attr("s") == 1) {
                    window.location.href = newsConfig.NewsStory + '?id=' + $(this).attr("id") + '&p=' + $(this).attr("p") + '&s=' + $(this).attr("source") + '&t=' + $(this).attr("t") + '&h=' + encodeURIComponent($($(this).find('.nd-news-headline-row')[0]).html());
                }
                else {
                    id = decodeURIComponent($(this).attr("id"));
                    var relatedInstrumentsJSON;
                    try {
                        relatedInstrumentsJSON = jQuery.parseJSON(id.substring(id.indexOf("{"), id.length));
                    }
                    catch (exception) {

                    }
                    if (relatedInstrumentsJSON) {
                        _showRelatedInstrumentsQL($(this), relatedInstrumentsJSON);
                    }
                }

            });

            jshtml = null;
        }
        else {
            return jshtml;
        }
    }


    var _renderNews = function (renderTo, groupByDate) {

        var _fields = {
            t: { name: 'Time', value: '', width: 65, hdTDcls: '' + _theme, TDCls: '' + _theme, cellCls: 'nd-news-timestamp' + _theme, hdCellCls: 'nd_fq_row' + _theme },
            h: { name: 'Headline', value: '', width: 182, hdTDcls: '' + _theme, TDCls: '' + _theme, cellCls: 'nd-news-headline' + _theme, hdCellCls: 'nd_fq_row' + _theme }


        };
        var _newsGrid = {
            renderers: {

                h: { fn: function (grid, rec, cellid, val, oldval, fxy) {
                    var t = rec.data.t;
                    var dateUnf = new Date(t * 1000);
                    t = ''; // '<span class="nd-headline-time nd-left">' + Netdania.Util.dateFormat(dateUnf, "HH:MM:ss").toLowerCase() + '</span>';

                    var storyCls = '';
                    if (rec.data.s != 1) {
                        storyCls = ' class="missingStory"';
                    }
                    return t + '<span ' + (rec.data.s == 1 ? 'style="cursor: pointer;"' : '') + storyCls + '>' + val + '</span>';
                }
                },
                t: { fn: function (grid, rec, cellid, val, oldval, fxy) {
                    if (val === '' || val === "N/A" || val === undefined) {
                        return "";
                    }
                    var dateUnf = new Date(val * 1000);

                    return Netdania.Util.dateFormat(dateUnf, "HH:MM").toLowerCase();
                    //'<span>' + Netdania.Util.dateFormat(dateUnf, "dd.mm.yy").toLowerCase() + '</span><br/>' +

                }
                }
            },
            records: [],
            columns: { t: 0, s: 0, h: 0 },
            provider: config.provider,
            fields: _fields,
            contentClass: 'nd_fq_rowNews' + _theme,
            contentOddClass: 'nd_fq_rowOddNews' + _theme,
            containerTopID: config.containerTopId,
            renderTo: config.containerId,
            groupByDate: false,
            hideHeaders: true,
            start: config.start,
            end: config.end,
            animate: config.animate || false,
            animateBgColor: config.animateBgColor || { first: '#648CC7', second: '#eeee' },
            animateForeColor: config.animateForeColor || { first: '#fff', second: '#000' },
            storyRender: config.storyRender,
            renderer: config.renderer,
            headlineMaxLength: config.headlineMaxLength
        };

        if (!_isLocalDS) {


            //check if the source needs the user be autheticated(login=true)
            var _needsLogin = false;
            var _canSourceBeUsed = true;
            var _user = new Netdania.Util.User();
            //console.log(config.source);
            Netdania.Util.overlayMask($('#' + config.containerId));
            if (config.start !== undefined && config.end !== undefined) {
                for (var i = 0; i < config.source.length; i++) {

                    //check if the source needs the user be autheticated(login=true)
                    _needsLogin = config.source[i].login || false;
                    _canSourceBeUsed = _needsLogin ? (_user.isAuthenticated() ? true : false) : true;
                    //end check
                    if (_canSourceBeUsed) {
                        var newsRequest = Netdania.JsApi.Request.getReqObjNewsSearch(config.source[i].source + (config.isin ? '#' + config.isin : ''), '', config.maxHeadlines, config.start, config.end, 0, config.source[i].provider);
                        _connection.AddRequest(newsRequest);
                        _requestIds.push(newsRequest.i);
                        var openFQ = (config.source[i].openFQ !== undefined) ? config.source[i].openFQ : (Netdania.News.Utilities.getOpenFQSource(config.source[i].source) !== '') ? Netdania.News.Utilities.getOpenFQSource(config.source[i].source) : undefined;
                        _requestInfo.push({ id: newsRequest.i, source: config.source[i].source, provider: config.source[i].provider, openFQ: openFQ });
                    }
                }
            }
            else {
                for (var i = 0; i < config.source.length; i++) {
                    //check if the source needs the user be autheticated(login=true)
                    _needsLogin = config.source[i].login || false;
                    //for the sources that are used without login is possible to be removed when the user has logged in.
                    //ex. if FXWire realtime is used when the user is logged in and FXWire_dl delayed when the user is not logged in, then is no 
                    //reason to use both realtime and delayed from fx wire when the user is logged in - so FXWire_dl will have useIfLoggedIn: false
                    _useIfUserIsLoggedIn = config.source[i].useIfLoggedIn || false;

                    _canSourceBeUsed = _needsLogin ? (_user.isAuthenticated() ? true : false) : (_user.isAuthenticated() ? _useIfUserIsLoggedIn : true);
                    //end check
                    if (_canSourceBeUsed) {

                        var newsRequest = Netdania.JsApi.Request.getReqObjHeadlines(config.source[i].source + (config.isin ? '#' + config.isin : ''), config.maxHeadlines, config.source[i].provider, config.isMonitor);
                        _connection.AddRequest(newsRequest);
                        _requestIds.push(newsRequest.i);
                        var openFQ = (config.source[i].openFQ !== undefined) ? config.source[i].openFQ : (Netdania.News.Utilities.getOpenFQSource(config.source[i].source) !== '') ? Netdania.News.Utilities.getOpenFQSource(config.source[i].source) : undefined;
                        _requestInfo.push({ id: newsRequest.i, source: config.source[i].source, provider: config.source[i].provider, openFQ: openFQ });
                    }
                }
            }
            if (_requestIds.length < 1) {
                _self.fireEvent(_self.NONEWSEVENT);
            } else {
                _connection.Flush();
            }


            _newsGrid.requestIds = _requestIds;


            _connection.addListener(Netdania.JsApi.Events.ONHISTORICALHEADLINES, function (newsHistoryResponse) {
                if (Netdania.Util.isInArray(newsHistoryResponse.id, _requestIds)) {

                    _requestIds = $.grep(_requestIds, function (value) {
                        return value != newsHistoryResponse.id;
                    });

                    if (newsHistoryResponse.data !== null) {
                        _cbfUpdateNews(newsHistoryResponse.data, newsHistoryResponse.id, newsHistoryResponse.type, _newsGrid);
                        if (_newsGrid.renderer !== undefined)
                            _newsGrid.renderer(_newsGrid);
                        else {
                            _renderNewsTable(_newsGrid);
                        }
                        if (config.pagination) {
                            _addPagination(grid);
                        }
                    }

                    if (_requestIds.length == 0) {
                        _connection.fireEvent(_self.NEWSRECEIVEDEVENT, [newsHistoryResponse.data, _newsGrid, newsHistoryResponse.id]);
                    }

                }

            });

            _connection.addListener(Netdania.JsApi.Events.ONHEADLINEUPDATE, function (monitorNewsResponse) {
                if (monitorNewsResponse.data !== null) {
                    var _updateData = [];
                    _updateData.push(monitorNewsResponse.data);
                    pagination.addNewInterval = false;
                    if (!config.pagination || (config.pagination && pagination.pageNo == 1)) {
                        _cbfUpdateNews(_updateData, monitorNewsResponse.id, monitorNewsResponse.type, _newsGrid);
                        if (Netdania.Util.isInArray(newsHistoryResponse.id, _newsGrid.requestIds)) {
                            if (_newsGrid.renderer !== undefined)
                                _newsGrid.renderer(_newsGrid);
                            else {
                                _renderNewsTable(_newsGrid);
                            }
                        }
                    }
                    if (config.pagination) {
                        _addPagination(_newsGrid);
                    }
                }
            });


            _connection.addListener(Netdania.JsApi.Events.ONNEWSSEARCH, function (newsHistoryResponse) {
                if (Netdania.Util.isInArray(newsHistoryResponse.id, _requestIds)) {

                    _requestIds = $.grep(_requestIds, function (value) {
                        return value != newsHistoryResponse.id;
                    });


                    if (newsHistoryResponse.data !== null) {
                        _cbfUpdateNews(newsHistoryResponse.data, newsHistoryResponse.id, newsHistoryResponse.type, _newsGrid);
                        _renderNews();
                    }
                    if (_requestIds.length == 0) {
                        _connection.fireEvent(_self.NEWSRECEIVEDEVENT, [newsHistoryResponse.data, _newsGrid, newsHistoryResponse.id]);
                    }
                }
            });


            _connection.addListener(_self.NEWSRECEIVEDEVENT, function (data, grid, id) {
                if (grid.records.length == 0) {
                    _self.fireEvent(_self.NONEWSEVENT, undefined);
                    if (config.defaultSource && config.isin !== '') {
                        config.isin = '';
                        config.source = config.defaultSource;
                        _renderNews();
                    }
                    else {
                        $('#pagination .next').hide();
                    }
                }
                else {
                    if (Netdania.Util.isInArray(id, _newsGrid.requestIds)) {
                        if (grid.renderer !== undefined)
                            grid.renderer(grid);
                        else {
                            _renderNewsTable(grid);
                        }
                        if (config.pagination) {
                            _addPagination(grid);
                        }
                    }
                }
                Netdania.Util.overlayMask($('#' + config.containerId), 'hide');
            });


        }
        else {
            return _cbfUpdateNews(_localDS.data, _localDS.id, _localDS.type, _newsGrid);
        }
    }

    var _requestNewsStory = function (newsId, provider, source) {

        Netdania.Util.overlayMask($('#' + config.containerId));
        var reqStory = Netdania.JsApi.Request.getReqObjStory(newsId, provider);
        _connection.AddRequest(reqStory);
        _connection.Flush();

        _connection.addListener(Netdania.JsApi.Events.ONNEWSSTORY, function (newsStoryResponse) {
            //raw json data received from the server
            //_logResponse(config.containerId, newsStoryResponse.data, "", newsStoryResponse.id, newsStoryResponse.type);

            Netdania.Util.overlayMask($('#' + config.containerId), 'hide');
            var _storyContent = newsStoryResponse.data;
            if (_source === 'RBB_StockExchange') {
                _storyContent = '<pre>' + _storyContent + '</pre>';
            }

            _storyContent = '<div class="nd-news-popup' + _theme + '">' + _storyContent + '</div>';

            //            $.modal(_storyContent, {
            //                escClose: true, overlayClose: true,
            //                position: [0, 0],
            //                modal: true, containerCss: (_theme === '' ? { 'background-color': '#E9E9F5', 'border': '4px solid #ccc', 'height': '500px', 'width': '550px'} : { 'height': '500px', 'width': '550px' })
            //            });

            $('body').append('<div id="nd-news-popup-' + config.containerId + '" class="nd-news-popup' + _theme + '">' + _storyContent + '</div>');
            var _storyDialog = $("#nd-news-popup-" + config.containerId).dialog({
                height: 480,
                width: 520,
                modal: true,
                close: function () {
                    $("#nd-news-popup-" + config.containerId).remove();
                },
                position: { my: "top", at: "top", of: '#nd-fq' },
                closeOnEscape: true,
                dialogClass: 'nd-fq-dialog'
            });

            //            _storyDialog.bind('clickoutside', function () {
            //                _storyDialog.dialog('close');
            //                $('#nd-news-popup-' + config.containerId).remove();
            //            });

            _connection.removeListener(Netdania.JsApi.Events.ONNEWSSTORY);
        });
    }

    var reqIds = [];
    var _showRelatedInstrumentsQL = function (element, relatedInstrumentsJSON) {
        connection.RemoveRequests(reqIds);
        $('.nd-newsStory-section').remove();
        if (element.attr('ql') !== "1" & relatedInstrumentsJSON.Syms.length > 0) {
            element.next().after('<tr><td colspan="2"><div id="nd-news-relatedInstruments-QL" class="nd-newsStory-section nd-news-relatedInstruments-QL" ></div></td></tr>');
            $('.nd_fq_rowNews').attr('ql', 0);
            element.attr('ql', 1);
            var configQL = {
                containerId: 'nd-news-relatedInstruments-QL',
                relatedInstrumentsJSON: relatedInstrumentsJSON
            };
            var relatedInstrQL = new Netdania.News.QuoteList.RelatedInstrumentsQL(configQL);
            reqIds = relatedInstrQL.init();
        }
        else {
            $('.nd_fq_rowNews').attr('ql', 0);
        }
    }

    var _getSourceProvider = function (id) {
        for (var i = 0; i < _requestInfo.length; i++)
            if (_requestInfo[i].id == id) {
                return { source: _requestInfo[i].source, provider: _requestInfo[i].provider, openFQ: _requestInfo[i].openFQ };
            }
        return {};
    }

    var _cbfUpdateNews = function (data, requestId, requestType, grid) {

        var _headlineObject, _r;
        var _eventObj = null;
        for (var lj = data.length - 1; lj >= 0; lj--) {
            _eventObj = null;
            _headlineObject = data[lj];
            var _sourceProvider = _getSourceProvider(requestId);
            if (_sourceProvider.source == undefined)
                _sourceProvider.source = config.source;
            if (_sourceProvider.provider == undefined)
                _sourceProvider.provider = config.provider;
            try {
                _eventObj = JSON.parse(_headlineObject.h);
            } catch (ex) {

            } finally {
            }

            if (_eventObj === null) {
                _r = {
                    id: "recid" + lj,
                    data: {
                        t: _headlineObject.t,
                        s: _headlineObject.s,
                        i: _headlineObject.i,
                        h: _headlineObject.h,
                        source: _sourceProvider.source,
                        provider: _sourceProvider.provider,
                        openFQ: _sourceProvider.openFQ
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
                    id: "recid" + lj,
                    data: {
                        t: _headlineObject.t,
                        s: _headlineObject.s,
                        i: _headlineObject.i,
                        h: _headlineText,
                        source: _sourceProvider.source,
                        provider: _sourceProvider.provider,
                        openFQ: _sourceProvider.openFQ
                    }
                };
            }

            grid.records.push(_r);

        }


        grid.records = grid.records.sort(function (a, b) {

            var _aT = a.data.t;
            var _bT = b.data.t;

            return _bT - _aT;
        });

        if (_removeNewsOnUpdate || grid.records.length > _itemsOnPage) {
            var extraRecords = grid.records.length - _itemsOnPage;
            for (var i = 0; i < extraRecords; i++)
                grid.records.pop();
        }

    }


    var _addPagination = function (grid) {


        $('#' + config.containerId).append('<div id="pagination"></div>');
        var instr = grid.records;
        if (pagination.addNewInterval !== false) {
            if (instr[instr.length - 1].data.t == (pagination.intervals[pagination.intervals.length - 1] - 1))
                pagination.stop = true;
            else
                pagination.intervals.push(instr[Math.min(instr.length, _itemsOnPage) - 1].data.t - 1);
            pagination.pageNo = pagination.intervals.length;
        }

        var cookies = Netdania.Util.Cookies();
        cookies.setCookie("pagination", JSON.stringify(pagination), 365);

        $('#pagination').pagination({
            pages: pagination.intervals.length,
            nextText: (pagination.stop || instr.length < _itemsOnPage) ? "" : "See more",
            selectOnClick: true,
            currentPage: pagination.pageNo,
            onPageClick: function (pageNumber) {
                pagination.addNewInterval = false;
                config.start = pagination.intervals[pageNumber - 1];
                config.end = (pagination.intervals[pageNumber - 2] === undefined) ? Math.round(new Date().getTime() / 1000) : pagination.intervals[pageNumber - 2];
                pagination.pageNo = pageNumber;
                _connection.removeListener(Netdania.JsApi.Events.ONHEADLINEUPDATE);
                _connection.removeListener(Netdania.JsApi.Events.ONHISTORICALHEADLINES);
                _connection.removeListener(Netdania.JsApi.Events.ONNEWSSEARCH);
                _connection.removeListener(_self.NEWSRECEIVEDEVENT);
                _renderNews();
            },
            onNextClick: function () {
                pagination.addNewInterval = true;
                config.start = 0;
                config.end = pagination.intervals[pagination.intervals.length - 1];
                _connection.removeListener(Netdania.JsApi.Events.ONHEADLINEUPDATE);
                _connection.removeListener(Netdania.JsApi.Events.ONHISTORICALHEADLINES);
                _connection.removeListener(Netdania.JsApi.Events.ONNEWSSEARCH);
                _connection.removeListener(_self.NEWSRECEIVEDEVENT);
                _renderNews();
            }
        });
        $('#pagination').addClass('nd-news-table');
    }

}
