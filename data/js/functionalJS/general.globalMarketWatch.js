var Netdania = Netdania || {};
Netdania.General = Netdania.General || {};

var globalCurrentCmp = globalCurrentCmp || 0;
var cbfunctions = cbfunctions || [];
var removeGlobalMWRequests = [];

Netdania.General.GlobalMarketWatch = function (containerId, dataSource) {

    this.init = function () {
        var host = Netdania.General.Config.Host; //"http://balancer11.netdania.com/StreamingServer/StreamingServer"; //the NetDania server to connect to

        var location = Netdania.Util.getLocation(Netdania.JsApi.Utilities.GetURL());
        var connectionParams = Netdania.General.Config.GetConnectionParams(location.pathname, Netdania.General.Config.URLConnectionParams);
        //console.log(connectionParams.pollingInterval, connectionParams.ct, 'market watch');
        var connectionConfig = {
            pollingInterval: connectionParams.pollingInterval, //milliseconds
            host: host,
            behavior: connectionParams.ct, //Netdania.JsApi.ConnectionType.LONGPOLLING, //LONG POLLING
            type: 1, //the server response format: JSONP
            v: 2
        }

        connection = new Netdania.JsApi.JSONConnection(connectionConfig);
        //connection.Connect();

        _createGlobalMarketWatchTabs();

    }


    var _createGlobalMarketWatchTabs = function () {

        var html = '';
        html += '<div id="nd-globalMW-header" ><table border="0" cellpadding="0" cellspacing="0" width="100%"><tr><td>' +
                    '<div id="nd-globalMW-tabs" class="nd-globalMW-tabs">' +
                        '<ul id="nd-globalMW-tabs-ul" class="nd-tabs-ul">' +
        '<li class="active"><a ref="#nd-globalMW-tab-global" type="globalMW_global" id="globalMW_global-a">Global</a></li>' +
                            '<li><a ref="#nd-globalMW-tab-forex" type="globalMW_forex" id="globalMW_forex-a">Forex</a></li>' +
                            '<li class="nd-tabs-lastTab"><span id="nd-globalMW-tabs-li-right"></span></li>' +
                        '</ul>' +
                    '</div></td></tr><tr><td>';

        html += '<div class="tab" id="nd-globalMW-tab-global" >' +
                    '<div id="nd-globalMW-tab-global-table" ></div>' +
                    '<div class="nd-globalMW-more"><span id="nd-globalMW-moreGlobal"><b>More</b></span></div>' +
                '</div>';
        html += '<div class="tab" id="nd-globalMW-tab-forex">' +
                    '<div id="nd-globalMW-tab-forex-table" ></div>' +
                    '<div class="nd-globalMW-more"><span id="nd-globalMW-moreForex"><b>More</b></span></div>' +
                '</div>';
        html += '</td></tr></table></div>';

        $('#' + containerId).html(html);

        _renderGlobalComponents('nd-globalMW-tab-global-table');
        // 

        $('#nd-globalMW-moreForex').click(function () {
            document.location.href = Netdania.General.Config.QuoteListURL + '?t=majors';
        });

        $('#nd-globalMW-moreGlobal').click(function () {
            document.location.href = Netdania.General.Config.QuoteListURL + '?&t=worldmarkets';
        });

        $('#globalMW_forex-a').click(function () { _renderForexComponents('nd-globalMW-tab-forex-table'); setActive($(this), "nd-globalMW-tabs-li-right"); });
        $('#globalMW_global-a').click(function () { _renderGlobalComponents('nd-globalMW-tab-global-table'); setActive($(this), "nd-globalMW-tabs-li-right"); });

        setActive($('.nd-globalMW-tabs').find('a[type="globalMW_global"]'), "nd-globalMW-tabs-li-right");
    }

    var _renderGlobalComponents = function (containerId) {
        if (removeGlobalMWRequests)
            connection.removeRequests(removeGlobalMWRequests);
        var config = {
            instruments: [{ s: "!DJI", n: "Dow Jones", p: "idc_rtb", d: 2 }
                            , { s: "UKX", n: "FTSE 100", p: "idc_dla", d: 2 }
                            , { s: "DAX.de", n: "DAX 30", p: "idc_dla", d: 2 }
                            , { s: "N225", n: "Nikkei 225", p: "idc_dla", d: 2 }
                            , { s: "I*NIFTY.in", n: "Nifty 50", p: "idc_dlc", d: 2 }
                            , { s: "XAUUSD", n: "Gold", p: "netdania_rt", d: 2 }
                            , { s: "EB0Y.ip", n: "Brent Crude", p: "idc_dla", d: 2}],
            containerId: containerId,
            fields: {
                f25: { name: 'Name', value: '', width: '30%', hdTDcls: ' ', TDCls: 'TDClsLeftMW', cellCls: 'nameColumnMW', hdCellCls: '' },
                f6: { name: 'Last', value: '', width: '27%', hdTDcls: '', TDCls: 'TDClsRightMW', cellCls: 'kursColumnMW', hdCellCls: '' },
                f14: { name: 'Change', value: '', width: '20%', hdTDcls: '', TDCls: 'TDClsRightMW', cellCls: 'changeColumnMW', hdCellCls: '' },
                f15: { name: '%Change', value: '', width: '23%', hdTDcls: '', TDCls: 'TDClsRightMW', cellCls: 'changeColumnMW', hdCellCls: '' }
            },
            removeRequests: true,
            dataSource: dataSource
        };
        _renderQuoteList(config);
    }

    var _renderForexComponents = function (containerId) {
        if (removeGlobalMWRequests)
            connection.removeRequests(removeGlobalMWRequests);
        var config = {
            instruments: [{ s: 'EURUSD', p: 'netdania_fxa', d: 5, n: 'EUR/USD' }
                            , { s: 'USDJPY', p: 'netdania_fxa', d: 3, n: 'USD/JPY' }
                            , { s: 'GBPUSD', p: 'netdania_fxa', d: 5, n: 'GBP/USD' }
                            , { s: 'AUDUSD', p: 'netdania_fxa', d: 5, n: 'AUD/USD' }
                            , { s: 'USDCAD', p: 'netdania_fxa', d: 5, n: 'USD/CAD' }
                            , { s: 'EURJPY', p: 'netdania_fxa', d: 5, n: 'EUR/JPY' }
                            , { s: 'EURGBP', p: 'netdania_fxa', d: 5, n: 'EUR/GBP'}],
            containerId: containerId,
            fields: {
                f25: { name: 'Name', value: '', width: '22%', hdTDcls: ' ', TDCls: 'TDClsLeftMW', cellCls: 'nameColumnMW', hdCellCls: '' },
                f6: { name: 'Last', value: '', width: '31%', hdTDcls: '', TDCls: 'TDClsRightMW', cellCls: 'kursColumnMW', hdCellCls: '' },
                f14: { name: 'Change', value: '', width: '25%', hdTDcls: '', TDCls: 'TDClsRightMW', cellCls: 'changeColumnMW', hdCellCls: '' },
                f15: { name: '%Change', value: '', width: '23%', hdTDcls: '', TDCls: 'TDClsRightMW', cellCls: 'changeColumnMW', hdCellCls: '' }
            },
            nameLength: 10,
            removeRequests: true,
            dataSource: dataSource
        };
        _renderQuoteList(config);
    }


    var _renderQuoteList = function (quotesConfig) {

        if (quotesConfig.dataSource) {
            var jsonDataSource = JSON.parse(quotesConfig.dataSource);
        }

        var customRenderer = {

            f25: function (grid, rec, cellid, val, oldval, fxy) {
                var realSymbol = rec.realSymbol;
                if (rec.realSymbol === undefined) {
                    realSymbol = rec.symbol;
                }

                if (val === undefined) {
                    val = "";
                } else {
                    if (rec.newName.f25 !== '')
                        val = rec.newName.f25;
                }

                return '<div class="nd-stocks-nameColumn" title="' + val + '" >' + Netdania.Util.stripellipsis(val, quotesConfig.nameLength) + '</div>'; // +marketInfo;
            },
            f14: function (grid, rec, cellid, val, oldval, fxy) {
                SetDataSourceForInstrument(jsonDataSource, rec.symbol, rec.provider, fxy, val);
                val = rec.decimals ? parseFloat(val).toFixed(rec.decimals) : val;

                if (val === "NaN") {
                    return '<div>Wait...</div>';
                }

                var valTrimmed = Netdania.removeLastDecimalPoint(val.indexOf(".") !== -1 ? val.replace(/0*$/, '') : val);

                var value = '<div>' + valTrimmed + '</div>';

                if (val > 0) {
                    value = '<div class="val-rise-MW">' + valTrimmed + '</div>';
                } else if (val < 0) {
                    value = '<div class="val-fall-MW">' + valTrimmed + '</div>';
                }
                return value;
            },

            f6: function (grid, rec, cellid, val, oldval, fxy) {
                var value = "";
                SetDataSourceForInstrument(jsonDataSource, rec.symbol, rec.provider, fxy, val);
                if (val.indexOf('/') === -1) {

                    val = rec.decimals ? parseFloat(val).toFixed(rec.decimals) : val;

                    if (val === "NaN") {
                        return '<div>Wait...</div>';
                    }


                    var lastTrimmed = val; // (val.indexOf(".") !== -1 && val.split('.').length > 1 ? val.replace(/0*$/, '') : val);

                    value = '<div>' + lastTrimmed + '</div>';
                    if (val > oldval) {
                        value = '<div class="val-rise-MW val-highlight">' + lastTrimmed + '</div>';
                    } else if (val < oldval) {
                        value = '<div class="val-fall-MW val-highlight">' + lastTrimmed + '</div>';
                    }
                }
                else {
                    var bidAskArr = val.split('/');

                    var bid = rec.bid;
                    var ask = rec.ask;

                    var oldBid = rec.oldBid;
                    var oldAsk = rec.oldAsk;

                    bid = rec.decimals ? parseFloat(bid).toFixed(rec.decimals) : bid;
                    ask = rec.decimals ? parseFloat(ask).toFixed(rec.decimals) : ask;

                    if (bid === "NaN") {
                        return '<div>Wait...</div>';
                    }

                    var bidTrimmed = ((bid.indexOf(".") !== -1 && parseFloat(bid.split('.')[1])) !== 0 ? bid.replace(/0*$/, '') : bid);
                    var askTrimmed = ((bidAskArr[1].indexOf(".") !== -1 && parseFloat(bidAskArr[1].split('.')[1])) !== 0 ? bidAskArr[1].replace(/0*$/, '') : bidAskArr[1]);
                    askTrimmed = askTrimmed.length > 1 ? askTrimmed : askTrimmed + '0';

                    var bidValue = '<span>' + bidTrimmed + '</span>';
                    var askValue = '<span>' + askTrimmed + '</span>';

                    if (bid > oldBid) {
                        bidValue = '<span class="val-rise-MW val-highlight">' + bidTrimmed + '</span>';
                    } else if (bid < oldBid) {
                        bidValue = '<span class="val-fall-MW val-highlight">' + bidTrimmed + '</span>';
                    }

                    if (ask > oldAsk) {
                        askValue = '<span class="val-rise-MW val-highlight">' + askTrimmed + '</span>';
                    } else if (ask < oldAsk) {
                        askValue = '<span class="val-fall-MW val-highlight">' + askTrimmed + '</span>';
                    }

                    value = bidValue + "/" + askValue;
                }

                return value;
            },
            f15: function (grid, rec, cellid, val, oldval, fxy) {
                SetDataSourceForInstrument(jsonDataSource, rec.symbol, rec.provider, fxy, val);
                var upImagePath = "/JsComponents/jsgeneral/images/up.gif";
                var downImagePath = "/JsComponents/jsgeneral/images/down.gif";
                val = parseFloat(val).toFixed(2);
                val = val.replace("NaN", "");
                if (val === "") {
                    return '<div></div>';
                }
                val = val.replace("-0.00", "0.00");
                var value = '<div>' + val.toString() + '%&nbsp; </div>';
                if (val > 0) {
                    value = '<div class="val-rise-MW">' + val + '%&nbsp;</div>';
                } else if (val < 0) {
                    value = '<div class="val-fall-MW">' + val + '%&nbsp;</div>';
                }
                return value;
            }
        };

        globalCurrentCmp++;
        var provider = "netdania_fxa";
        var useDefaultColumns = quotesConfig.columns || true;
        var instr = [];


        for (var i = 0; i < quotesConfig.instruments.length; i++) {
            var data = GetDataSourceForInstrument(jsonDataSource, quotesConfig.instruments[i].s, quotesConfig.instruments[i].p);
            if (data !== undefined && data["f6"] == "N/A" && data["f10"] !== "N/A" && data["f11"] !== "N/A") {
                data["f6"] = getBidAsk(data["f10"], data["f11"], quotesConfig.instruments[i].d || 5);
            }
            instr.push({ r: Netdania.JsApi.Request.getReqObjPrice(quotesConfig.instruments[i].s, quotesConfig.instruments[i].p || provider, true),
                f25: quotesConfig.instruments[i].n || '',
                d: quotesConfig.instruments[i].d,
                data: (data !== undefined) ? data : {}
            });
        }

        var cmpMarketWatch = { records: [], instr: {}, arr: [], arrRemove: [] };

        cmpMarketWatch.renderers = {
            f25: {
                fn: customRenderer.f25
            },
            f6: {
                fn: customRenderer.f6
                , hightlightInterval: 1000
            },
            f14: {
                fn: customRenderer.f14
            },
            f15: {
                fn: customRenderer.f15
            }
        };

        removeGlobalMWRequests = [];
        for (var ii = 0; ii < instr.length; ii++) {
            cmpMarketWatch.records.push({
                id: 'recid-' + instr[ii].r.i,
                symbol: instr[ii].r.s,
                decimals: instr[ii].d,
                data: instr[ii].data,
                newName: { f25: instr[ii].f25 },
                provider: instr[ii].r.p,
                market: instr[ii].m,
                delay: instr[ii].delay,
                bid: (instr[ii].data !== undefined && instr[ii].data["f10"] !== undefined) ? instr[ii].data["f10"] : undefined,
                ask: (instr[ii].data !== undefined && instr[ii].data["f11"] !== undefined) ? instr[ii].data["f11"] : undefined
            });
            cmpMarketWatch.arr.push(instr[ii].r);
            cmpMarketWatch.arrRemove.push(Netdania.JsApi.Request.getReqObjRemove(instr[ii].r.i));

            if (quotesConfig.removeRequests)
                removeGlobalMWRequests.push(instr[ii].r.i);
        }

        connection.addListener(Netdania.JsApi.Events.ONRAWUPDATE, function (data, ts, id, type) {
            //for (var kk = 0; kk < cbfunctions.length; kk++) {
            //    if (cbfunctions[kk].hasOwnProperty('win')) {
            //        if (cbfunctions[kk].hasOwnProperty('win') !== null) {
            //            cbfunctions[kk].update(data, cbfunctions[kk].cmpMarketWatchMarketWatch, cbfunctions[kk].win);
            //        }
            //    }
            //    else {
            //        cbfunctions[kk].update(data, cbfunctions[kk].cmpMarketWatch);
            //    }
            //}
            receiveQuotesJsonData(data, cmpMarketWatch);
        });

        connection.appendRequests(cmpMarketWatch.arr);
        connection.Flush();
        cmpMarketWatch.instr = instr;

        //cbfunctions.push({
        //    update: receiveQuotesJsonData,
        //    cmp: cmpMarketWatch
        //});

        if (quotesConfig.fields) {
            fields = quotesConfig.fields;
        }
        else {
            fields = {
                f25: { name: 'Name', value: '', width: '25%', hdTDcls: ' ', TDCls: 'TDClsLeftMW', cellCls: 'nameColumnMW', hdCellCls: '' },
                f6: { name: 'Last', value: '', width: '28%', hdTDcls: '', TDCls: 'TDClsRightMW', cellCls: 'kursColumnMW', hdCellCls: '' },
                f14: { name: 'Change', value: '', width: '25%', hdTDcls: '', TDCls: 'TDClsRightMW', cellCls: 'changeColumnMW', hdCellCls: '' },
                f15: { name: '%Change', value: '', width: '22%', hdTDcls: '', TDCls: 'TDClsRightMW', cellCls: 'changeColumnMW', hdCellCls: '' }
            };
        }

        cmpMarketWatch.fields = fields;
        var tableConfig = {
            instr: cmpMarketWatch,
            fields: fields,
            contentClass: 'globalDefaultContentClass',
            contentOddClass: 'globalDefaultContentOddClass',
            width: '100%',
            renderTo: quotesConfig.containerId,
            height: '100%',
            headerCls: 'headCls',
            hideHeaders: true
        };
        return _renderTable(tableConfig);
    }

    var GetDataSourceForInstrument = function (dataSource, symbol, provider) {
        if (dataSource !== undefined) {
            for (var i = 0; i < dataSource.length; i++) {
                if (dataSource[i].s == symbol && dataSource[i].p == provider) {
                    return dataSource[i].data;
                }
            }
        }
        return undefined;
    }

    var SetDataSourceForInstrument = function (dataSource, symbol, provider, field, value) {
        if (dataSource !== undefined) {
            for (var i = 0; i < dataSource.length; i++) {
                if (dataSource[i].s == symbol && dataSource[i].p == provider) {
                    dataSource[i].data[field] = value;
                }
            }
        }
    }

    var _showFQ = function (symbol, provider, decimals) {
        var _query = '?symbol={0}&provider={1}&decimals={2}&ext=false';
        document.location.href = Netdania.General.Config.FullQuoteURL + _query.replace('{0}', symbol).replace('{1}', provider).replace('{2}', decimals);
    }

    var _renderTable = function (tableConfig) {
        var headerClass, contentClass, contentOddClass, width, fields, instr;

        fields = tableConfig.fields;
        var cmp = tableConfig.instr;
        instr = cmp.records;
        //console.log(instr);
        var tableWidth = 0;
        var headerHtml = '';
        var colsHtml = '';
        var lastField = '';
        contentClass = tableConfig.contentClass;
        contentOddClass = tableConfig.contentOddClass;


        if (tableConfig.width !== undefined && tableConfig.width !== null) {
            tableWidth = tableConfig.width;
        }
        for (var prop in fields) {
            lastField = prop;
        }
        var cols = 0;
        for (var prop in fields) {
            if (fields.hasOwnProperty(prop)) {
                cols++;
                //colsHtml += '<col width='+fields[prop].width+'>';
                //tableWidth += parseInt(fields[prop].width);
                var lc = '';
                //                if (lastField === prop) {
                //                    lc = ' lastColCls';
                //                }
                headerHtml += '<th width="' + fields[prop].width + '" class="' + fields[prop].hdTDcls + lc + '"><div class="' + fields[prop].hdCellCls + '"><div>' +
                         fields[prop].name + '</div></div></th>';
            }
        }

        var forceHeight = '';
        if (tableConfig.height !== undefined) {
            forceHeight = 'style="overflow: auto;overflow-x: hidden; height: ' + tableConfig.height + 'px;"';
        }

        var align = '';
        if (tableConfig.align === '') {
            align = "center";
        }
        else {
            align = tableConfig.align;
        }
        var jshtml = '';

        jshtml = '<table style="table-layout:fixed" width=' + tableWidth + '  border="0" cellspacing="0" cellpadding="0">';

        jshtml += colsHtml;
        //var t = jshtml;
        if (tableConfig.hideHeaders !== true) {
            var headerCls = '';
            if (tableConfig.headerCls !== undefined) {
                headerCls = 'class="' + tableConfig.headerCls + '"';
            }
            jshtml += '<thead ' + headerCls + '><tr>';
            jshtml += headerHtml;
            jshtml += '</tr></thead>';
        }

        jshtml += '<tbody>';
        var gday = -1;
        var gmonth = -1;
        var gyear = -1;
        if (tableConfig.groupByDate === true && instr.length > 0) {
            gd = new Date(instr[0].data.t * 1000);
        }

        var noDataMessage = tableConfig.noDataMessage;
        if (noDataMessage === undefined) {
            noDataMessage = 'No data available!';
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
            if (ii % 2 === 0) {
                jshtml += '<tr style="cursor:pointer" s="' + instr[ii].symbol + '" p="' + instr[ii].provider + '" d="' + instr[ii].decimals + '" class="' + contentClass + '">';
            }
            else {
                jshtml += '<tr style="cursor:pointer" s="' + instr[ii].symbol + '" p="' + instr[ii].provider + '" d="' + instr[ii].decimals + '"  class="' + contentOddClass + '">';
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
                        value = cmp.renderers[prop1].fn(cmp, instr[ii], cellid, value, value, prop1);
                    } else {
                        if (value === undefined || value === '') value = 'N/A';
                        value = '<div>' + value + '</div>';
                    }
                    var lc = '';
                    //                    if (lastField === prop1) {
                    //                        lc = ' lastColCls';
                    //                    }
                    jshtml += '<td width="' + fields[prop1].width + '" nowrap style="overflow: hidden;" class="' + fields[prop1].TDCls + lc + '"><div id="' + cellid + '" class="' + fields[prop1].cellCls + '" >' + value + '</div></td>';
                }
            }
            jshtml += '</tr>';
        }
        jshtml += '</tbody>';
        jshtml += '</table>';

        if (tableConfig.renderTo === 'script') {
            return jshtml;
        }
        if (tableConfig.renderTo !== undefined) {
            $('#' + tableConfig.renderTo).html(jshtml);
            $('#' + tableConfig.renderTo + ' tr').click(function () {
                _showFQ($(this).attr("s"), $(this).attr("p"), $(this).attr("d"));
            });
        }
        else {
            window.document.write(jshtml);
        }


    }


    this.init();
}

//function removeGlobalRequests() {
//    console.log(removeGlobalMWRequests);
//    var host = Netdania.General.Config.Host; //"http://balancer11.netdania.com/StreamingServer/StreamingServer"; //the NetDania server to connect to
//    connectionConfig = {
//        pollingInterval: 1000, //milliseconds
//        host: host,
//        behavior: Netdania.JsApi.ConnectionType.LONGPOLLING, //LONG POLLING
//        type: 1 //the server response format: JSONP
//    }

//    connection = new Netdania.JsApi.JSONConnection(connectionConfig);
//    connection.Connect();

//    if (removeGlobalMWRequests)
//        connection.removeRequests(removeGlobalMWRequests);
//}


