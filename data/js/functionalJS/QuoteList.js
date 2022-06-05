Netdania = Netdania || {};
Netdania.Alert = Netdania.Alert || {};

Netdania.Alert.QuoteList = function (config) {
    // Events.enable.call(this);
    // var _self = this;
    var _mainRequestId;
    var _symbol = config.symbol.split('|');
    var _provider = config.provider;
    var _decimals = config.decimals;
    var _showAddButton = config.showAddButton;
    var theme = '';
    var _selectedObject = [];
    var _callbackFunction = config.callbackFunction;
    var selectedField = config.selectedField;
    var updateField = config.updateField;
    var initialUpdate = config.initialUpdate;
    var operatorID = config.operatorID;
    var ddHtmlElementID = config.ddHtmlElementID;
    this.getSelectedObject = function () {
        return _selectedObject;
    }
    var fieldValueUse;
    var instrumentsHiLow = [];

    var addSelectedObject = function (element) {
        for (var i = 0; i < _selectedObject.length; i++) {
            if (_selectedObject[i].f == element.f) {
                _selectedObject[i] = element;
                return;
            }
        }
        _selectedObject.push(element);
    }

    /*
    Compare the High and Low containing fields (1WEEK HIGH/LOW, 1MONTH HIGH/LOW  with the HIGH/LOW values
    If the value is higher/lower it returns the HIGH/LOW values
     */
    var HiLowCompare = function (val) {

        var lowField = ["f3"];
        var highField = ["f2"];
        var lowerFields = ["f121", "f123", "f125", "f127"];
        var higherFields = ["f120", "f122", "f124", "f126"];

        var instrument = val.instrument;
        var field = val.updateField;
        var value = val.updateValue;

        /*
         Update the instrument lower fields
         */
        var updateLowerFields = function () {
            try {
                for (var i = 0; i < lowerFields.length; i++) {
                    if (instrument.data != undefined && instrument.data[lowerFields[i]] != undefined) {
                        if (instrument.low != undefined) {
                            if (parseFloat(instrument.data[lowerFields[i]]) > instrument.low) {
                                instrument.data[lowerFields[i]] = instrument.low;
                            }
                        }
                    }
                }
            }
            catch (err) {

                Netdania.Utilities.LogError("QuoteList.HiLowCompare.updateLowerFields", err);
            }
        }

        /*
         Update the instrument higher fields
         */
        var updateHigherFields = function () {
            try {
                for (var i = 0; i < higherFields.length; i++) {
                    if (instrument.data != undefined && instrument.data[higherFields[i]] != undefined) {
                        if (instrument.high != undefined) {
                            if (parseFloat(instrument.data[higherFields[i]]) < instrument.high) {
                                instrument.data[higherFields[i]] = instrument.high;
                            }
                        }
                    }
                }
            }
            catch (err) {

                Netdania.Utilities.LogError("QuoteList.HiLowCompare.updateHigherFields", err);
            }
        }

        try {

            /*
             LOW Field - update the Low value for this instrument
             */
            for (var i = 0; i < lowField.length; i++) {
                if (lowField[i] == field) {
                    instrument.low = parseFloat(value);
                    updateLowerFields();
                    return value;
                }
            }

            /*
             HIGH Field - update the High value for this instrument
             */
            for (var i = 0; i < highField.length; i++) {
                if (highField[i] == field) {
                    instrument.high = parseFloat(value);
                    updateHigherFields();
                    return value;
                }
            }

            /*
             LOW contains fields
             eg. 1WEEk LOW is bigger than the LOW we return the LOW value
             */
            for (var i = 0; i < lowerFields.length; i++) {
                if (lowerFields[i] == field) {
                    if (instrument.low != undefined) {
                        if (parseFloat(value) > instrument.low) {
                            return instrument.low;
                        }
                    }
                }
            }

            /*
             HIGH contains fields
             eg. 1WEEk HIGH is lower than the HIGH we return the HIGH value
             */
            for (var i = 0; i < higherFields.length; i++) {
                if (higherFields[i] == field) {
                    if (instrument.high != undefined) {
                        if (parseFloat(value) < instrument.high) {
                            return instrument.high;
                        }
                    }
                }
            }

        }
        catch (err) {

            Netdania.Utilities.LogError("QuoteList.HiLowCompare", err);
        }
        return value
    }


    var defaultRenderer = {
        f25: {
            fn: function (grid, rec, cellid, val, oldval, fxy) {
                var realSymbol = rec.realSymbol;
                if (rec.realSymbol === undefined) {
                    realSymbol = rec.symbol;
                }

                if (val == undefined) {
                    val = "";
                } else if (rec.newName.f25 != undefined) {
                    val = rec.newName.f25;
                }

                return '<span title="' + val + '">' + Netdania.Util.stripHeadline(val, 30) + '</span>';

            }
        },

        fAdd: {
            fn: function (grid, rec, cellid, val, oldval, fxy) {

                if (_showAddButton) {
                    return '<image class="nd-alert-add-button" src="/JSComponents/Alerts/images/bell.png" title="Add alert" alt="' + rec.symbol + '|' + rec.provider + '"></image>';
                }
                else {
                    return '';
                }
            }
        },
        f27: {
            fn: function (grid, rec, cellid, val, oldval, fxy) {
                if (val !== 'N/A' && val !== '') {

                     val = HiLowCompare({instrument:rec, updateField:fxy, updateValue:val});

                    if (rec.decimals === undefined) {
                        val = parseFloat(val).toFixed(2);
                    } else {
                        val = parseFloat(val).toFixed(rec.decimals);
                    }

                    if (!isNaN(val)) {

                        if (fieldValueUse == fxy && initialUpdate)
                        {
                            $('#' + updateField).val(val);
                            $('#' + updateField).change();
                            initialUpdate = false;
                        }

                        var value = '<span class="value' + theme + '">' + val + '</span>';
                        if (val > 0) {
                            value = '<span class="upValue' + theme + '">' + val + '</span>';
                        } else if (val < 0) {
                            value = '<span class="downValue' + theme + '">' + val + '</span>';
                        }
                        return value;
                    }
                    else
                        return '<span class="value' + theme + '" >N/A</span>';
                }
                else
                    return '<span class="value' + theme + '" >N/A</span>';


            }
        }
    }

    this.init = function () {
        var host = Netdania.General.Config.Host; //the NetDania server to connect to
        var connectionConfig = {
            pollingInterval: Netdania.JsApi.PoolingInterval.AUTO, //milliseconds
            host: host,
            behavior: Netdania.JsApi.ConnectionType.AUTO, //LONG POLLING
            type: 1 //the server response format: JSONP
        }
        //create a json connection object
        connection = new Netdania.JsApi.JSONConnection(connectionConfig);
        //connection.Connect();
    }

    this.addRequests = function (cmp, instrumentsArray) {

        var instr = [];
        for (var i = 0; i < instrumentsArray.length; i++) {
            var InstrumentSymbol = instrumentsArray[i].symbol;
            var InstrumentProvider = instrumentsArray[i].provider;
            var InstrumentDecimals = instrumentsArray[i].decimals;

            var reqId = Netdania.JsApi.Request.getReqObjPrice(InstrumentSymbol, InstrumentProvider, true);
            instr.push({
                r: reqId,
                decimals: InstrumentDecimals
            });
        }

        for (var i = 0; i < instr.length; i++) {
            cmp.records.push({
                id: 'recid-' + instr[i].r.i,
                symbol: instr[i].r.s,
                data: {},
                newName: { f25: instr[i].f25 },
                provider: instr[i].r.p,
                decimals: instr[i].decimals
            });

            connection.AddRequest(instr[i].r);
            connection.Flush();
        }

        cmp.instr = instr;
        connection.addListener(Netdania.JsApi.Events.ONPRICEUPDATE, function (data) {
            for (var i = 0; i < instr.length; i++) {
                if (instr[i].r.i == data.id) {
                    var instrumentRealType = Netdania.Util.getIntrumentRealType(data);
                    if (_callbackFunction) {
                        if (instrumentRealType >= 0) {
                            connection.removeListener(Netdania.JsApi.Events.ONPRICEUPDATE);
                            _callbackFunction(instrumentRealType, selectedField);
                        }
                    }
                    var  fields = {
                        f25: { name: 'Name', value: '', width: '110px', hdTDcls: 'hdTDcls', TDCls: 'TDCls', cellCls: 'nameColumn', hdCellCls: 'nameHeaderClass' }
                        , f10: { name: 'Bid', value: '', width: '70px', hdTDcls: 'hdTDclsLeftAlert', TDCls: 'TDClsLeftAlert', cellCls: 'kursColumn', hdCellCls: 'gridDefaultHeader' }
                        , f11: { name: 'Ask', value: '', width: '70px', hdTDcls: 'hdTDclsLeftAlert', TDCls: 'TDClsLeftAlert', cellCls: 'kursColumn', hdCellCls: 'gridDefaultHeader' }
                        , f6: { name: 'Last', value: '', width: '70px', hdTDcls: 'hdTDclsLeftAlert', TDCls: 'TDClsLeftAlert', cellCls: 'kursColumn', hdCellCls: 'gridDefaultHeader' }
                        , f2: { name: 'High', value: '', width: '70px', hdTDcls: 'hdTDclsLeftAlert', TDCls: 'TDClsLeftAlert', cellCls: 'kursColumn', hdCellCls: 'gridDefaultHeader', condition: '>'}
                        , f3: { name: 'Low', value: '', width: '70px', hdTDcls: 'hdTDclsLeftAlert', TDCls: 'TDClsLeftAlert', cellCls: 'kursColumn', hdCellCls: 'gridDefaultHeader', condition: '<'}
                        , f120: { name: '1 Week High', value: '', width: '40%', hdTDcls: 'hdTDclsRightAlert', TDCls: 'TDClsRightAlert', cellCls: 'kursColumn', hdCellCls: 'gridDefaultHeader' ,showinDD: true, condition: '>'}
                        , f121: { name: '1 Week Low', value: '', width: '40%', hdTDcls: 'hdTDclsRightAlert', TDCls: 'TDClsRightAlert', cellCls: 'kursColumn', hdCellCls: 'gridDefaultHeader' ,showinDD: true, condition: '<'}
                        , f122: { name: '1 Month High', value: '', width: '40%', hdTDcls: 'hdTDclsRightAlert', TDCls: 'TDClsRightAlert', cellCls: 'kursColumn', hdCellCls: 'gridDefaultHeader' ,showinDD: true, condition: '>'}
                        , f123: { name: '1 Month Low', value: '', width: '40%', hdTDcls: 'hdTDclsRightAlert', TDCls: 'TDClsRightAlert', cellCls: 'kursColumn', hdCellCls: 'gridDefaultHeader' ,showinDD: true, condition: '<'}
                        , f124: { name: '3 Months High', value: '', width: '40%', hdTDcls: 'hdTDclsRightAlert', TDCls: 'TDClsRightAlert', cellCls: 'kursColumn', hdCellCls: 'gridDefaultHeader' ,showinDD: true, condition: '>'}
                        , f125: { name: '3 Months Low', value: '', width: '40%', hdTDcls: 'hdTDclsRightAlert', TDCls: 'TDClsRightAlert', cellCls: 'kursColumn', hdCellCls: 'gridDefaultHeader' ,showinDD: true, condition: '<'}
                        , f126: { name: '6 Months High', value: '', width: '40%', hdTDcls: 'hdTDclsRightAlert', TDCls: 'TDClsRightAlert', cellCls: 'kursColumn', hdCellCls: 'gridDefaultHeader' ,showinDD: true, condition: '>'}
                        , f127: { name: '6 Months Low', value: '', width: '40%', hdTDcls: 'hdTDclsRightAlert', TDCls: 'TDClsRightAlert', cellCls: 'kursColumn', hdCellCls: 'gridDefaultHeader' ,showinDD: true, condition: '<'}
                        , f21: { name: '52 Weeks High', value: '', width: '40%', hdTDcls: 'hdTDclsRightAlert', TDCls: 'TDClsRightAlert', cellCls: 'kursColumn', hdCellCls: 'gridDefaultHeader' ,showinDD: true, condition: '>'}
                        , f22: { name: '52 Weeks Low', value: '', width: '40%', hdTDcls: 'hdTDclsRightAlert', TDCls: 'TDClsRightAlert', cellCls: 'kursColumn', hdCellCls: 'gridDefaultHeader' ,showinDD: true, condition: '<'}
                    };

                    //the field from which we take compare value
                    fieldValueUse = "f10";

                    switch (instrumentRealType) {
                        case 0:
                            {
                                //remove LAST
                                   if (fields.f6) delete fields.f6;
                                break;

                            }
                        case 1:
                            {
                              //remove BID anad ASK
                                if (fields.f10) delete fields.f10;
                                if (fields.f11) delete fields.f11;
                                fieldValueUse = "f6";
                                break;
                            }
                        case 2:
                            {
                               //all fields nothing to remove
                                break;
                            }
                    }

                    //configure quotes table
                    var tableConfig = {
                        instr: cmp,
                        fields: fields,
                        //  contentClass:'nd_fq_row' + theme,
                        //  contentOddClass:'nd_fq_rowOdd' + theme,
                        renderTo: 'nd-instrumentBidAsk',
                        hideHeaders: false,
                        width: ''
                    };
                    renderTable(tableConfig);
                }
            }

        });
        connection.addListener(Netdania.JsApi.Events.ONRAWUPDATE, function (data, ts, id, type) {
            update(data, cmp);
        });

        return cmp;
    }

    this.renderTable = function (windowId, instruments, isHeaderVisible, isCompetitorsTable) {

        window[windowId] = {
            id: windowId,
            renderers: {
                f25: defaultRenderer.f25,
                f10: defaultRenderer.f27,
                f11: defaultRenderer.f27,
                f6: defaultRenderer.f27,
                f2: defaultRenderer.f27,
                f3: defaultRenderer.f27,
                f120: defaultRenderer.f27,
                f121: defaultRenderer.f27,
                f122: defaultRenderer.f27,
                f123: defaultRenderer.f27,
                f124: defaultRenderer.f27,
                f125: defaultRenderer.f27,
                f126: defaultRenderer.f27,
                f127: defaultRenderer.f27,
                f21: defaultRenderer.f27,
                f22: defaultRenderer.f27,
                fAdd: defaultRenderer.fAdd

            },
            records: [],
            columns: { f25: 0, f10: 0, f11: 0, f6: 0, f2: 0, f3: 0, f120: 0, f121: 0, f122: 0, f123: 0, f124: 0, f125: 0, f126: 0, f127: 0, f21: 0, f22: 0 }
        };

        var cmp = this.addRequests(window[windowId], instruments);
    }


    //render the quotes table
    function renderTable(tableConfig, startFrame, endFrame) {

        var headerClass, contentClass, contentOddClass, width, fields, instr;

        fields = tableConfig.fields;
        //filter only fields without showinDD parameter
       /* for (var prop in fields)
        {
            if (fields[prop].showinDD) {delete fields[prop];}
        }*/


        var cmp = tableConfig.instr;
        instr = cmp.records;
        var tableWidth = 0;
        var headerHtml = '';
        var colsHtml = '';
        var lastField = '';
        var ddDisplayField = '';
        contentClass = tableConfig.contentClass;
        if (tableConfig.contentClass === undefined || tableConfig.contentClass === null || tableConfig.contentClass === '') {
            contentClass = "gridDefaultContentClassAlert";
        }
        contentOddClass = tableConfig.contentOddClass;
        if (tableConfig.contentOddClass === undefined || tableConfig.contentOddClass === null || tableConfig.contentOddClass === '') {
            contentOddClass = "gridDefaultContentOddClassAlert";
        }

        if (tableConfig.width !== undefined && tableConfig.width !== null) {
            tableWidth = tableConfig.width;
        }
        for (var prop in fields) {
            lastField = prop;
        }
        var cols = 0;

        /*
        for (var prop in fields) {
            if (fields[prop].showinDD)
            {ddDisplayField = prop; lastField = ddDisplayField; break;}
        }
        */

        var recid = instr[0].id;
        var instrumentsHtml = '<table class="nd-alert-table-options">';
        for (var prop1 in fields) {
            if (fields[prop1].showinDD && prop1 != ddDisplayField)
            {

                var cellid = recid + '-' + prop1;
                var value = instr[0].data[prop1];
                if (value === undefined) {
                    value = '';
                }
                if (cmp.renderers[prop1] !== undefined) {
                    value = cmp.renderers[prop1].fn(cmp, instr[0], cellid, value, value, prop1);
                } else {
                    value = '<span>' + value + '</span>';
                }



                var lc = '';
                if (lastField === prop1) {
                    lc = ' lastColCls';
                }
                instrumentsHtml += ' <tr class="nd-alert-table-option" condition="' +  fields[prop1].condition + '" title="Click to add field to Alert Value"><td>' +
                    ' <label class="nd-fq-alert-option-symbol">' +  fields[prop1].name + '</label></td><td>' +
                    ' <label class="nd-fq-alert-option-text" id="' + cellid + '">' + value + '</label></td></tr>';
            }

        }
        instrumentsHtml += '</table>';

        var ddHtml = '<div class="nd-alert-field-select"><a class="nd-alert-selected">Other fields</a><span class="nd-alert-field-pointer nd-alert-field-pointer-down"></span>' + instrumentsHtml + '</div>';



        for (var prop in fields) {
            if (fields.hasOwnProperty(prop)) {
                if (!fields[prop].showinDD)
                {
                    cols++;
                    var lc = '';
                    if (lastField === prop) {
                        lc = ' lastColCls';
                    }
                    headerHtml += '<th width="' + fields[prop].width + '" class="' + fields[prop].hdTDcls + lc + '"><span class="' + fields[prop].hdCellCls + '"><span>' + fields[prop].name + '</span></span></th>';
                }
            }
        }
       // headerHtml += '<th rowspan="2" width="100px" class="hdTDclsRightAlert lastColCls" style="padding-left:20px;"><span class=kursColumn">'+ddHtml+'</span></th>';

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
        var jshtml = '<table style="table-layout:fixed" width=' + tableWidth + '  border="0" cellspacing="0" cellpadding="0">';
        jshtml += colsHtml;

        if (tableConfig.hideHeaders !== true) {
            jshtml += '<thead><tr>';
            jshtml += headerHtml;
            jshtml += '</tr></thead>';
        }

        jshtml += '<tbody >';
        var gday = -1;
        var gmonth = -1;
        var gyear = -1;
        if (tableConfig.groupByDate === true && instr.length > 0) {
            gd = new Date(instr[0].data.t * 1000);
        }

        var noDataMessage = tableConfig.noDataMessage;
        if (noDataMessage === undefined) {
            noDataMessage = 'No updates available!';
        }
        if (instr.length == 0) {
            jshtml += '<tr><td colspan=' + cols + ' class="netd-grid-empty">' + noDataMessage + '</td></tr>';
        }
        for (var ii = 0; ii < instr.length; ii++) {
            if (tableConfig.groupByDate === true) {
                var cd = new Date(instr[ii].data.t * 1000);
                if (cd.getDay() !== gday || cd.getMonth() !== gmonth || cd.getYear() !== gyear) {
                    jshtml += '<tr><td colspan=' + cols + ' class="netd-group-hd ">' + dateFormat(cd, "dd. mmmm yyyy") + '</td></tr>';
                    gday = cd.getDay();
                    gmonth = cd.getMonth();
                    gyear = cd.getYear();
                }
            }
            var style = (tableConfig.isCompetitorsTable) ? 'style = "cursor:pointer"' : '';
            if (ii % 2 === 0) {
                jshtml += '<tr class="' + contentClass + '" symbol ="' + instr[ii].symbol + '" provider ="' + instr[ii].provider + '" ' + style + ' >';
            }
            else {
                jshtml += '<tr class="' + contentOddClass + '" symbol ="' + instr[ii].symbol + '" provider ="' + instr[ii].provider + '" ' + style + '>';
            }

            var recid = instr[ii].id;
            var counter = 0;
            for (var prop1 in fields) {
                if (fields.hasOwnProperty(prop1)) {
                    if (!fields[prop1].showinDD)
                    {
                    var cellid = recid + '-' + prop1;
                    var value = instr[ii].data[prop1];
                    if (value === undefined) {
                        value = '';
                    }
                    if (cmp.renderers[prop1] !== undefined) {
                        value = cmp.renderers[prop1].fn(cmp, instr[ii], cellid, value, value, prop1);
                    } else {
                        value = '<span>' + value + '</span>';
                    }
                    var lc = '';
                    if (lastField === prop1) {
                        lc = ' lastColCls';
                    }
                    jshtml += '<td title="Click to add field to Alert Value"  style="overflow: hidden;" class="' + fields[prop1].TDCls + lc + '"><span id="' + cellid + '" class="' + fields[prop1].cellCls + '" >' + value + '</span></td>';
                    }
                }
                counter++;
                if (counter == 8) jshtml += '</tr><tr>';
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
        }
        else {
            window.document.write(jshtml);
        }

        if (ddHtmlElementID !== undefined) {
            var el = document.getElementById(ddHtmlElementID);
            replaceHtml(el, ddHtml);
        }

        $('.nd-alert-table-options').css('right',  0);
        $('.nd-alert-field-select').click(function () {
            $('.nd-alert-table-options').toggle();
        });
        $('.nd-alert-table-options').mouseleave(function () {
           $('.nd-alert-table-options').hide();
        });
        $('.nd-alert-table-option').click(function () {
            var value = $(this).children(':nth-child(2)').children(':nth-child(1)').children(':nth-child(1)').html();
            var condition = $(this).attr('condition');
            $( '#' + operatorID).val(condition);
            $( '#' + operatorID).change();
            $( '#' + updateField).val(value);
            $( '#' + updateField).change();
        });


        $('.kursColumn').click(function () {

            var value = $(this).children(':nth-child(1)').html();

            $( '#' + updateField).val(value);
            $( '#' + updateField).change();
        });

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
    };

    function update(data, grid) {
        var recs = grid.records;
        var validCmp = false;
        if (grid != undefined && grid.instr != undefined && grid.instr.length > 0 && grid.instr[0].r != undefined) {
            addSelectedObject({ f: 'p', v: grid.instr[0].r.p });
            addSelectedObject({ f: 's', v: grid.instr[0].r.s });
        }

        for (var k = 0; k < data.length; k++) {
            if (data[k].f) {
                var r = getRecordById(grid.records, "recid-" + data[k].i);
                if (r === undefined) {
                    continue;
                }
                validCmp = true;
                var last = "notdirty";
                var bid = "notdirty";
                var ask = "notdirty";
                var close = "notdirty";
                var yclose = "notdirty";
                var name = "notdirty";
                var name2 = "notdirty";
                var realSymbol = "notdirty";
                var tempOldValue = '';
                var tempOldName = r.data.f25;

                for (var i = 0; data[k] && i < data[k].f.length; i++) {
                    var field = data[k].f[i];
                    if (field.f === 6) {
                        r.last = parseFloat(field.v).toFixed(4);
                        last = "dirty";
                    }
                    if (field.f === 10) {
                        r.bid = field.v;
                        bid = 'dirty';
                    }
                    if (field.f === 11) {
                        r.ask = field.v;
                        ask = 'dirty';
                    }
                    if (field.f === 1) {
                        r.close = field.v;
                        close = 'dirty';
                    }
                    if (field.f === 19) {
                        r.yclose = field.v;
                        yclose = 'dirty';
                    }
                    if (field.f === 1003) {
                        if (r.realSymbol !== undefined) {
                            realSymbol = "dirty";
                        }
                        r.realSymbol = field.v;
                    }
                    if (field.f === 39) {
                        r.isin = field.v;
                    }

                    if ("f" + field.f in grid.columns) {
                        addSelectedObject(field);
                        tempOldValue = r.data["f" + field.f];
                        r.data["f" + field.f] = field.v;
                        if (field.f === 6) {
                            r.data.f6 = r.last;
                        }
                        if (field.f === 25) {
                            if (tempOldValue === undefined) {
                                r.data.f25 = field.v;
                                name = "dirty";
                            }
                        } else if (field.f === 5013) {
                            r.data.f5013 = field.v;
                            name2 = "dirty";
                        } else {
                            refreshCell(grid, r, r.id + '-f' + field.f, field.v, tempOldValue, 'f' + field.f);
                        }
                    }

                }
                if (r.realSymbol !== undefined) {
                    last = "dirty";
                }
                if (name === "dirty" || realSymbol === "dirty") {
                    refreshCell(grid, r, r.id + '-f25', r.data.f25, tempOldName, 'f25');
                }
                if (name2 === "dirty") {
                    refreshCell(grid, r, r.id + '-f5013', r.data.f5013, '', 'f5013');
                }
                if (grid.contentType === 'indexes' && last === "notdirty" && (bid === "dirty" || ask === "dirty")) {
                    r.last = getLast("N/A", r.bid, r.ask, 4);
                    last = "dirty";
                    if ('f6' in grid.columns) {
                        tempOldValue = r.data.f6;
                        r.data.f6 = r.last;
                        refreshCell(grid, r, r.id + '-f6', r.last, tempOldValue, 'f6');
                    }
                }

                if (last !== "notdirty" || close !== "notdirty") {
                    if ('f15' in grid.columns) {
                        tempOldValue = r.data.f15;
                        r.data.f15 = computePctChange(r.last, r.close);
                        refreshCell(grid, r, r.id + '-f15', r.data.f15, tempOldValue, 'f15');
                    }
                    if ('f14' in grid.columns) {
                        tempOldValue = r.data.f14;
                        r.data.f14 = computeChange(r.last, r.close);
                        refreshCell(grid, r, r.id + '-f14', r.data.f14, tempOldValue, 'f14');
                    }
                }

                if (last !== "notdirty" || yclose !== "notdirty") {
                    if ('f32' in grid.columns) {
                        tempOldValue = r.data.f32;
                        r.data.f32 = computePctYChange(r.last, r.yclose);
                        refreshCell(grid, r, r.id + '-f32', r.data.f32, tempOldValue, 'f32');
                    }
                    if ('f31' in grid.columns) {
                        tempOldValue = r.data.f31;
                        r.data.f31 = computeYChange(r.last, r.yclose);
                        refreshCell(grid, r, r.id + '-f31', r.data.f31, tempOldValue, 'f31');
                    }

                }
            }
        }

    }

    function computePctYChange(last, yclose) {
        if (yclose !== "" && yclose !== 0) {
            return Math.floor(parseFloat((last - yclose) * 100 / yclose) * 100) / 100;
        }
        return 'N/A';
    }


    function getRecordById(records, id) {
        for (var k = 0; k < records.length; k++) {
            if (records[k].id === id) {
                return records[k];
            }
        }
        return undefined;
    }


    function refreshCell(grid, rec, cellid, val, oldval, fxy) {
        var torender = '<span>' + val + '</span>';
        if (grid.renderers[fxy] !== undefined) {
            torender = grid.renderers[fxy].fn(grid, rec, cellid, val, oldval, fxy);
        }
        var cell = document.getElementById(cellid);

        if (cell !== null) {
            cell.innerHTML = torender;
        }
        if (grid.renderers[fxy] !== undefined && grid.renderers[fxy].hightlightInterval !== undefined) {
            setTimeout('reset_highlight("' + cellid + '")', grid.renderers[fxy].hightlightInterval);
        }
    }

    this.render = function () {

        this.init();
        var instrument = [];
        for (var i = 0; i < _symbol.length; i++) {

            instrument.push({symbol: _symbol[i], provider: _provider, decimals: _decimals})
        }
        this.renderTable(config.containerId, instrument, true);

    }



    this.render();
    return this;
};