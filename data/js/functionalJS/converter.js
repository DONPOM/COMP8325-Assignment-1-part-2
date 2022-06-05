var NetdaniaConverter = NetdaniaConverter || {};

//Control Call
//NetdaniaConverter.CurrencyConverter({from:'GBP', to:'EUR', defaultAmount:1, provider:'idc_lite', pairs:pairs, containerID:'converter'});

// Available currencies list which will be loaded into dropdownlist. 
// here you can add/remove currencies from the converter
// each currency has the folling fields:
//    s -  ISO code from currency
//    n -  CCY code (name) for currency
//    units - number of units from the first currency
//    decimals -  number of decimals for displaying the converted result and the rate. Optional field - If is not set, the defaultDecimals parameter will be used

NetdaniaConverter.pairs = [
    {s:"AED", n:"United Arab Emirates dirham"},
    {s:"AFN", n:"New Afghan afghani"},
    {s:"ALL", n:"Albanian lek"},
    {s:"AMD", n:"Armenian dram"},
    {s:"ANG", n:"Netherlands Antillean guilder"},
    {s:"AOA", n:"Angolan kwanza"},
    {s:"ARS", n:"Argentine peso"},
    {s:"AUD", n:"Australian dollar"},
    {s:"AWG", n:"Aruban florin"},
    {s:"AZN", n:"Azerbaijani manat"},
    {s:"BAM", n:"Bosnia and Herzegovina mark"},
    {s:"BBD", n:"Barbadian dollar"},
    {s:"BDT", n:"Bangladeshi taka"},
    {s:"BEF", n:"Belgian convertible franc"},
    {s:"BGN", n:"Bulgarian lev"},
    {s:"BHD", n:"Bahraini dinar"},
    {s:"BIF", n:"Burundian franc"},
    {s:"BMD", n:"Bermudian dollar"},
    {s:"BND", n:"Brunei dollar"},
    {s:"BOB", n:"Bolivian boliviano"},
    {s:"BRL", n:"Brazilian real"},
    {s:"BSD", n:"Bahamian dollar"},
    {s:"BTN", n:"Bhutanese ngultrum"},
    {s:"BWP", n:"Botswana pula"},
    {s:"BYR", n:"Belarusian ruble"},
    {s:"BZD", n:"Belize dollar"},
    {s:"CAD", n:"Canadian dollar"},
    {s:"CDF", n:"Congolese franc"},
    {s:"CHF", n:"Swiss franc"},
    {s:"CLP", n:"Chilean peso"},
    {s:"CNY", n:"Chinese yuan"},
    {s:"CNH", n:"Chinese Renminbi Off-Shore"},
    {s:"COP", n:"Colombian peso"},
    {s:"CRC", n:"Costa Rican colón"},
    {s:"CUC", n:"Cuban convertible peso"},
    {s:"CUP", n:"Cuban peso"},
    {s:"CVE", n:"Cape Verdean escudo"},
    {s:"CZK", n:"Czech koruna"},
    {s:"DJF", n:"Djiboutian franc"},
    {s:"DKK", n:"Danish krone"},
    {s:"DOP", n:"Dominican peso"},
    {s:"DZD", n:"Algerian dinar"},
    {s:"EGP", n:"Egyptian pound"},
    {s:"ERN", n:"Eritrean nakfa"},
    {s:"ETB", n:"Ethiopian birr"},
    {s:"EUR", n:"Euro"},
    {s:"FJD", n:"Fijian dollar"},
    {s:"FKP", n:"Falkland Islands pound"},
    {s:"GBP", n:"British pound"},
    {s:"GEL", n:"Georgian lari"},
    {s:"GHS", n:"Ghana cedi"},
    {s:"GIP", n:"Gibraltar pound"},
    {s:"GMD", n:"Gambian dalasi"},
    {s:"GNF", n:"Guinean franc"},
    {s:"GTQ", n:"Guatemalan quetzal"},
    {s:"GYD", n:"Guyanese dollar"},
    {s:"HKD", n:"Hong Kong dollar"},
    {s:"HNL", n:"Honduran lempira"},
    {s:"HRK", n:"Croatian kuna"},
    {s:"HTG", n:"Haitian gourde"},
    {s:"HUF", n:"Hungarian forint"},
    {s:"IDR", n:"Indonesian rupiah"},
    {s:"ILS", n:"Israeli new shekel"},
    {s:"INR", n:"Indian rupee"},
    {s:"IQD", n:"Iraqi dinar"},
    {s:"IRR", n:"Iranian rial", decimals:8},
    {s:"ISK", n:"Icelandic króna"},
    {s:"JMD", n:"Jamaican dollar"},
    {s:"JOD", n:"Jordanian dinar"},
    {s:"JPY", n:"Japanese yen"},
    {s:"KES", n:"Kenyan shilling"},
    {s:"KGS", n:"Kyrgyzstani som"},
    {s:"KHR", n:"Cambodian riel"},
    {s:"KMF", n:"Comorian franc"},
    {s:"KPW", n:"North Korean won"},
    {s:"KRW", n:"South Korean won"},
    {s:"KWD", n:"Kuwaiti dinar"},
    {s:"KYD", n:"Cayman Islands dollar"},
    {s:"KZT", n:"Kazakhstani tenge"},
    {s:"LAK", n:"Lao kip"},
    {s:"LBP", n:"Lebanese pound"},
    {s:"LKR", n:"Sri Lankan rupee"},
    {s:"LRD", n:"Liberian dollar"},
    {s:"LSL", n:"Lesotho loti"},
    {s:"LTL", n:"Lithuanian litas"},
    {s:"LYD", n:"Libyan dinar"},
    {s:"MAD", n:"Moroccan dirham"},
    {s:"MDL", n:"Moldovan leu"},
    {s:"MGA", n:"Malagasy ariary"},
    {s:"MKD", n:"Macedonian denar"},
    {s:"MMK", n:"Burmese kyat"},
    {s:"MNT", n:"Mongolian tögrög"},
    {s:"MOP", n:"Macanese pataca"},
    {s:"MRO", n:"Mauritanian ouguiya"},
    {s:"MUR", n:"Mauritian rupee"},
    {s:"MVR", n:"Maldivian rufiyaa"},
    {s:"MWK", n:"Malawian kwacha"},
    {s:"MXN", n:"Mexican peso"},
    {s:"MYR", n:"Malaysian ringgit"},
    {s:"NAD", n:"Namibian dollar"},
    {s:"NGN", n:"Nigerian naira"},
    {s:"NIO", n:"Nicaraguan córdoba"},
    {s:"NOK", n:"Norwegian krone"},
    {s:"NPR", n:"Nepalese rupee"},
    {s:"NZD", n:"New Zealand dollar"},
    {s:"OMR", n:"Omani rial"},
    {s:"PAB", n:"Panamanian balboa"},
    {s:"PEN", n:"Peruvian nuevo sol"},
    {s:"PGK", n:"Papua New Guinean kina"},
    {s:"PHP", n:"Philippine peso"},
    {s:"PKR", n:"Pakistani rupee"},
    {s:"PLN", n:"Polish zloty"},
    {s:"PYG", n:"Paraguayan guaraní"},
    {s:"QAR", n:"Qatari riyal"},
    {s:"RON", n:"Romanian leu"},
    {s:"RSD", n:"Serbian dinar"},
    {s:"RUB", n:"Russian ruble"},
    {s:"RWF", n:"Rwandan franc"},
    {s:"SAR", n:"Saudi riyal"},
    {s:"SBD", n:"Solomon Islands dollar"},
    {s:"SCR", n:"Seychellois rupee"},
    {s:"SDG", n:"Sudanese pound"},
    {s:"SGD", n:"Singapore dollar"},
    {s:"SEK", n:"Swedish krona"},
    {s:"SHP", n:"Saint Helena pound"},
    {s:"SLL", n:"Sierra Leonean leone"},
    {s:"SOS", n:"Somali shilling"},
    {s:"SRD", n:"Surinamese dollar"},
    {s:"STD", n:"São Tomé and Príncipe dobra"},
    {s:"SVC", n:"Salvadoran colón"},
    {s:"SYP", n:"Syrian pound"},
    {s:"SZL", n:"Swazi lilangeni"},
    {s:"THB", n:"Thai baht"},
    {s:"TJS", n:"Tajikistani somoni"},
    {s:"TMT", n:"Turkmenistan manat"},
    {s:"TND", n:"Tunisian dinar"},
    {s:"TOP", n:"Tongan paanga"},
    {s:"TRY", n:"Turkish lira"},
    {s:"TTD", n:"Trinidad and Tobago dollar"},
    {s:"TWD", n:"New Taiwan dollar"},
    {s:"TZS", n:"Tanzanian shilling"},
    {s:"UAH", n:"Ukrainian hryvnia"},
    {s:"UGX", n:"Ugandan shilling"},
    {s:"USD", n:"United States dollar"},
    {s:"UYU", n:"Uruguayan peso"},
    {s:"UZS", n:"Uzbekistani som"},
    {s:"VEF", n:"Venezuelan bolívar"},
    {s:"VND", n:"Vietnamese dong", decimals:8},
    {s:"VUV", n:"Vanuatu vatu"},
    {s:"WST", n:"Samoan tala"},
    {s:"XAF", n:"Central African CFA franc"},
    {s:"XCD", n:"IMF Special Drawing Rights"},
    {s:"XOF", n:"West African CFA franc"},
    {s:"XPF", n:"CFP franc"},
    {s:"YER", n:"Yemeni rial"},
    {s:"ZAR", n:"South African rand"},
    {s:"ZMW", n:"Zambian kwacha"}];
// defaultDecimals parameter will be used for displaying result and rate in case the currency from "TO" select list has no decimal field defined in currency list
NetdaniaConverter.defaultDecimals = 5;
// default pair can be set when calling convert function (eg. NetdaniaConverter.renderCurrencyConverter({from:'GBP',to:'EUR'});). If no parameter is set, defaultPair defined here id used
NetdaniaConverter.defaultPair = ['EUR', 'USD'];
// default amount can be set when calling convert function (eg. NetdaniaConverter.renderCurrencyConverter({defaultAmount:1});). If no parameter is set, defaultAmount defined here id used
NetdaniaConverter.defaultAmount = 1;
// error message displayed when letters or invalid symbols are entered.
NetdaniaConverter.amountErrorMessage = 'Positive numbers only!';
//Provider
NetdaniaConverter.provider = "idc_lite";
//Other currency to calculate cross rates if the the default pair is not working
NetdaniaConverter.CrossCurrencyRates = ['USD', 'GBP', 'EUR'];

NetdaniaConverter.updated = false;
NetdaniaConverter.hasTwoRates = false;
NetdaniaConverter.firstRate = 0;
NetdaniaConverter.idsRequest = [];
NetdaniaConverter.idsRequestInfo = [];
NetdaniaConverter.CrossCurrencyRatesRequested = []

NetdaniaConverter.CurrencyConverter = function (data) {

    var _self = this;
    var connection;
    this.init = function () {

        var host = Netdania.General.Config.Host;
        var connectionConfig = {
            pollingInterval: 1000, //milliseconds
            host: host,
            behavior: Netdania.JsApi.ConnectionType.POLLING, //LONG POLLING
            type: 1 //the server response format: JSONP
        }

        connection = new Netdania.JsApi.JSONConnection(connectionConfig);
       // connection.Connect();

        var queryFrom = Netdania.Util.queryString('ccfrom');
        var queryTo = Netdania.Util.queryString('ccto');

        var appFrom = data.from !== undefined ? data.from : NetdaniaConverter.defaultPair[0];
        var appTo = data.to !== undefined ? data.to : NetdaniaConverter.defaultPair[1];
        NetdaniaConverter.defaultPair[0] = (queryFrom != null && queryFrom !== undefined && queryFrom != '' && queryFrom != "false" ? queryFrom : appFrom);
        NetdaniaConverter.defaultPair[1] = (queryTo != null && queryTo !== undefined && queryTo != '' && queryTo != "false" ? queryTo : appTo);



        NetdaniaConverter.defaultAmount = data.defaultAmount !== undefined ? data.defaultAmount : NetdaniaConverter.defaultAmount

        this.id = Netdania.JsApi.globalCurrentReqId;


        NetdaniaConverter.provider = (data.provider !== undefined ? data.provider : NetdaniaConverter.provider);
        NetdaniaConverter.pairs = (data.pairs !== undefined ? data.pairs : NetdaniaConverter.pairs);

        var html = '<div class="nd-converterDiv">' +

            '<table class="nd-TableConverter">' +
            '<tr>' +
            '<td class="nd-TableConverter-firstCell" >' +
            '<div id="currencyFrom" >' +
            '</div>' +
            '</td>' +
            '<td >' +
            '<div id="invertCurrency" class="nd-TableConverter-middleCell" >' +
            '<input class="nd-switchBtn" type="button" value="" id="switchBtn' + this.id + '" onclick="NetdaniaConverter.SwitchCurrency();" /> ' +
            '</div>' +
            '</td>' +
            '<td class="nd-TableConverter-lastCell">' +
            '<div id="currencyTo" >' +
            '</div>' +
            '</td>' +
            '</tr>' +
            '<tr>' +
            '<td class="nd-TableConverter-firstCell" >' +
            '<div class="nd-resultClass">' +
            NetdaniaConverter.Converter_rendRate(this.id, "From") +
            '</div >' +
            '</td>' +
            '<td class="nd-TableConverter-middleCell">' +

            '</td>' +
            '<td class="nd-TableConverter-lastCell" >' +
            '<div class="nd-resultClass">' +
            NetdaniaConverter.Converter_rendRate(this.id, "To") +
            '</div >' +
            '</td>' +
            '</tr>' +
            '<tr>' +
            '<td class="nd-TableConverter-firstCell" >' +
            '<div>' + NetdaniaConverter.Converter_rendAmount(this.id, NetdaniaConverter.defaultAmount) + '</div>' +
            '</td>' +
            '<td class="nd-TableConverter-middleCell">' +

            '</td>' +
            '<td class="nd-TableConverter-lastCell">' +
            '<div>' + NetdaniaConverter.Converter_rendResult(this.id) + '</div>' +
            '</td>' +
            '</tr>' +
            '<tr>' +
            '<td colspan="2">' +
            '<div id="nd-errormess" class="nd-errormess"></div>' +
            '</td>' +
            '<td>' +
            '<div><input class="nd-convertBtn" type="button" value="Convert" id="convertBtn' + this.id + '" onclick="NetdaniaConverter.GetRate(\'' + this.id + '\');" ></div>' +
            '</td>' +
            '</tr>' +
            '</table>' +
            '</div>';

        if (data.containerID !== undefined) {
            $('#' + data.containerID).html(html);
        } else {
            window.document.write(html);
        }

        this.RenderCurrencyConverter(this.id, NetdaniaConverter.defaultAmount);
    }

    this._callbackSelectTo = function (_selectedObject) {
        lookupTo.selectedItem = _selectedObject;
        _self.GetRate(_self.id);
    }

    this._callbackSelectFrom = function (_selectedObject) {
        lookupFrom.selectedItem = _selectedObject;
        _self.GetRate(_self.id);
    }

    this.RenderCurrencyConverter = function (id, defaultAmount) {
        try {

            var lookupConfigCurrencyFrom = {
                parentName: "currencyFrom",
                inputName: "currencyFromInput",
                width: '95px',
                showButton: false,
                callback: this._callbackSelectFrom,
                noItemFound: "No item found!!!",
                typeSearch: "Type or select...",
                isCombobox: true,
                ConverterObject: NetdaniaConverter,
                defaultValue: "",
                showIcon: false
            }
            var lookupConfigCurrencyTo = {
                parentName: "currencyTo",
                inputName: "currencyToInput",
                width: '95px',
                showButton: false,
                callback: this._callbackSelectTo,
                noItemFound: "No item found!!!",
                typeSearch: "Type or select..",
                isCombobox: true,
                ConverterObject: NetdaniaConverter,
                defaultValue: "",
                showIcon: false
            }

            lookupFrom = new setLookupControl(lookupConfigCurrencyFrom);

            lookupTo = new setLookupControl(lookupConfigCurrencyTo);

            var itemFrom = NetdaniaConverter.GetItem(NetdaniaConverter.pairs, NetdaniaConverter.defaultPair[0]);
            if (itemFrom != undefined) {
                lookupFrom.Select({ item: { label: itemFrom.s, value: itemFrom.s} });
            }

            var itemTo = NetdaniaConverter.GetItem(NetdaniaConverter.pairs, NetdaniaConverter.defaultPair[1]);
            if (itemTo != undefined) {
                lookupTo.Select({ item: { label: itemTo.s, value: itemTo.s} });
            }

        } catch (err) {
            Netdania.Util.LogError("NetdaniaConverter.RenderCurrencyConverter", err);
        }
    }


    var CallBackFunctionGetRate = function (rate) {
        try {
            if (rate !== undefined) {
                if (rate.available) {
                    NetdaniaConverter.Converter_convert(rate.value, rate.cross, rate.decimals, rate.units);
                }
                else {
                    GetRateWithOther();
                }
            }
        } catch (err) {
            Netdania.Util.LogError("NetdaniaConverter.CallBackFunctionGetRate", err);
        }
    }

    this.GetRate = function (id) {
        try {
            $('#result' + id).val('Please wait...');
            $('#rate' + id + 'From').html('...');
            $('#rate' + id + 'To').html('...');


            if (lookupFrom.selectedItem !== undefined && lookupFrom.selectedItem.value != undefined && lookupFrom.selectedItem.value != '' && lookupTo.selectedItem !== undefined && lookupTo.selectedItem.value != undefined && lookupTo.selectedItem.value != '') {
                var valueSymbol = lookupFrom.selectedItem.value + lookupTo.selectedItem.value;
                NetdaniaConverter.CrossCurrencyRatesRequested = [];
                _self.GetPrice({ symbol: valueSymbol, callback: CallBackFunctionGetRate });
            }

        } catch (err) {
            Netdania.Util.LogError("NetdaniaConverter.GetRate", err);
        }
    }

    this.GetPrice = function (data) {

        try {

            var _getRequestData = function (id) {
                for (var i = 0; i < NetdaniaConverter.idsRequestInfo.length; i++)
                    if (NetdaniaConverter.idsRequestInfo[i].id == id) {
                        return NetdaniaConverter.idsRequestInfo[i];
                    }
                return {};
            }

            this.lastCall = data.symbol;

            var instr = Netdania.JsApi.Request.getReqObjPrice(data.symbol, NetdaniaConverter.provider, 0);
            NetdaniaConverter.idsRequest.push(Netdania.JsApi.globalCurrentReqId);
            NetdaniaConverter.idsRequestInfo.push({ id: Netdania.JsApi.globalCurrentReqId, symbol: data.symbol, data: data })

            connection.AddRequest(instr);
            connection.Flush();



            connection.addListener(Netdania.JsApi.Events.ONPRICEUPDATE, function (monitorPriceResponse) {
                try {
                    if (Netdania.Util.isInArray(monitorPriceResponse.id, NetdaniaConverter.idsRequest)) {
                        NetdaniaConverter.idsRequest = $.grep(NetdaniaConverter.idsRequest, function (value) {
                            return value != monitorPriceResponse.id;
                        });

                        var _request = _getRequestData(monitorPriceResponse.id);

                        if (monitorPriceResponse != null && monitorPriceResponse.data != null && monitorPriceResponse.data.length > 0) {
                            var priceValue = monitorPriceResponse.get(Netdania.JsApi.Fields.QUOTE_ASK);
                            if (priceValue != "N/A") {

                                var unitsValue = monitorPriceResponse.get(Netdania.JsApi.Fields.QUOTE_UNITS);
                                unitsValue = (unitsValue == "N/A" ? undefined : unitsValue);
                                var decimalsValue = monitorPriceResponse.get(Netdania.JsApi.Fields.QUOTE_DECIMALS);
                                decimalsValue = (decimalsValue == "N/A" ? undefined : decimalsValue);

                                _request.data.callback({ available: true, value: priceValue, type: _request.type, data: _request.data, decimals: decimalsValue, units: unitsValue });
                            }
                            else {
                                _request.data.callback({ available: false, value: undefined, data: _request.data });
                            }
                        }
                        else {
                            _request.data.callback({ available: false, value: undefined, data: _request.data });
                        }
                    }
                }
                catch (err) {
                    Netdania.Util.LogError("NetdaniaConverter.GetPrice.ONPRICEUPDATE", err);
                }

            });

        } catch (err) {
            Netdania.Util.LogError("NetdaniaConverter.GetPrice", err);
        }
    }


    var CallBackFunctionRateCalculate = function RateCalculate(request) {
        try {
            if (request.data.type == 0) {
                _self.FromCrossValue = request.value;
                _self.FromCrossValueReceived = true;

            }

            if (request.data.type == 1) {
                _self.ToCrossValue = request.value;
                _self.ToCrossValueReceived = true;
            }

            if (_self.ToCrossValueReceived !== undefined && _self.ToCrossValueReceived != false && _self.FromCrossValueReceived !== undefined && _self.FromCrossValueReceived != false) {

                if (_self.ToCrossValue != undefined && _self.FromCrossValue !== undefined) {
                    var rate = _self.ToCrossValue / _self.FromCrossValue
                    CallBackFunctionGetRate({ available: true, value: rate, cross: true });
                }
                else {
                    CallBackFunctionGetRate({ available: false });
                }
            }
        } catch (err) {
            Netdania.Util.LogError("NetdaniaConverter.GetRateWithOther", err);
        }
    }

    var GetRateWithOther = function (callBackFunction) {
        try {
            $.each(NetdaniaConverter.CrossCurrencyRates, function (index, RateSymbol) {
                if ($.inArray(RateSymbol, NetdaniaConverter.CrossCurrencyRatesRequested) == -1) {
                    NetdaniaConverter.CrossCurrencyRatesRequested.push(RateSymbol);

                    var _fromSymbol = lookupFrom.selectedItem.value;
                    var _toSymbol = lookupTo.selectedItem.value;
                    var _firstPair = RateSymbol + _fromSymbol;
                    var _secondPair = RateSymbol + _toSymbol;

                    _self.FromCrossValueReceived = false;
                    _self.ToCrossValueReceived = false;

                    _self.FromCrossValue = undefined;
                    _self.ToCrossValue = undefined;


                    _self.GetPrice({ symbol: _firstPair, callback: CallBackFunctionRateCalculate, type: 0, callbackOriginal: callBackFunction });
                    _self.GetPrice({ symbol: _secondPair, callback: CallBackFunctionRateCalculate, type: 1, callbackOriginal: callBackFunction });

                    return false;
                }
            });
        } catch (err) {
            Netdania.Util.LogError("NetdaniaConverter.GetRateWithOther", err);
        }
    }

    this.SwitchCurrency = function () {
        if (lookupFrom.selectedItem !== undefined && lookupFrom.selectedItem.value != undefined && lookupFrom.selectedItem.value != '' && lookupTo.selectedItem !== undefined && lookupTo.selectedItem.value != undefined && lookupTo.selectedItem.value != '') {
            var from = lookupFrom.selectedItem.value
            var to = lookupTo.selectedItem.value
            lookupTo.selectedItem = undefined;
            lookupFrom.Select({ item: { label: to, value: to} });
            lookupTo.Select({ item: { label: from, value: from} });
        }
    }

    this.GetItem = function (data, value) {
        for (i = 0; i < data.length; i++) {
            if (data[i].s == value) {
                return data[i];
            }
        }
    }

    this.init();

};


NetdaniaConverter.Converter_convert = function (publicRate, crossUsed, serverDecimals, serverUnits) {
    try {
        var id= NetdaniaConverter.id;
        $('#nd-errormess').html('');
        var amount = document.getElementById('amount' + id).value;

        var amountOK = true;
        try {
            amount = parseFloat(amount);
        }
        catch (ex) {
            amountOK = false;
            $('#nd-errormess').html(NetdaniaConverter.amountErrorMessage);
            $('#result' + id).html('...');
            $('#rate' + id).html('...&nbsp;');
        }

        if (amount === '' || amount < 0 || amountOK === false || isNaN(amount)) {
            $('#nd-errormess').html(NetdaniaConverter.amountErrorMessage);
            $('#result' + id).html('...');
            $('#rate' + id).html('...');
            return;
        }

        var result = document.getElementById('result' + id);
        var rate = document.getElementById('rate' + id + "From");
        var rateInvert = document.getElementById('rate' + id + "To");



        if (publicRate != 'N/A') {

            var fromDecimals = (serverDecimals !== undefined ? serverDecimals : NetdaniaConverter.defaultDecimals);
            var fromUnits = (serverUnits !==undefined ? Math.pow(10, serverUnits) :1);
            var itemFrom = NetdaniaConverter.GetItem(NetdaniaConverter.pairs, lookupFrom.selectedItem.value);
            if (itemFrom !== undefined) {

                if (crossUsed == null || crossUsed === undefined || crossUsed == false) {
                    pairUnitsFound = false;
                    /*Get the units from the pair if exist*/
                    if (itemFrom.pairUnits !== undefined) {
                        $.each(itemFrom.pairUnits, function (index, data) {
                            if (data.s == lookupTo.selectedItem.value) {
                                fromUnits = data.units;
                                pairUnitsFound = true;
                                return false;
                            }
                        });
                    }

                    if (!pairUnitsFound) {
                        /*If no pair found we take the default if defined*/
                        fromUnits = (itemFrom.units !== undefined ? itemFrom.units : fromUnits);
                    }
                }

                fromDecimals = (itemFrom.decimals !== undefined ? itemFrom.decimals : fromDecimals);
            }


            var toDecimals = NetdaniaConverter.defaultDecimals;
            var toUnits = 1;
            var itemTo = NetdaniaConverter.GetItem(NetdaniaConverter.pairs, lookupTo.selectedItem.value);
            if (itemTo !== undefined) {
                toUnits = (itemTo.units !== undefined ? itemTo.units : toUnits);
                toDecimals = (itemTo.decimals !== undefined ? itemTo.decimals : toDecimals);
            }

            publicRate = publicRate / fromUnits;
            rate.innerHTML = parseFloat(publicRate).toFixed(fromDecimals).replace('.', ',');

            rateInvert.innerHTML = (1 / parseFloat(publicRate)).toFixed(toDecimals).replace('.', ',');

            result.value = parseFloat(amount * publicRate).toFixed(fromDecimals).replace('.', ',');

        }
        else {
            //set this flag on false in order to try to get the value again
            NetdaniaConverter.updated = false;
            rate.innerHTML = "Wait...";
            rateInvert.innerHTML = "Wait...";
            result.value = "0.00";
        }
    } catch (err) {
        Netdania.Util.LogError("NetdaniaConverter.Converter_convert", err);
    }

}

NetdaniaConverter.Converter_rendAmount = function (id, defaultAmount) {
    var html = '<input class="nd-input_amount" onblur="this.value = this.value.replace(\'.\',\',\');" onclick="this.value=\'\';"  type="text" value="' + defaultAmount + '" MAXLENGTH="8" id="amount' + id + '"/>';
    return html;
}

NetdaniaConverter.changeText = function (select, currenciesArray) {
    NetdaniaConverter.resetDropDown(select, currenciesArray);
    select.options[select.selectedIndex].text = currenciesArray[select.selectedIndex].n;
}

NetdaniaConverter.resetDropDown = function (select, currenciesArray) {
    for (var i = 0; i < currenciesArray.length; i++) {
        select.options[i].text = currenciesArray[i].n;
    }
}

NetdaniaConverter.Converter_rendCurrency = function (arr, id, selectedValue, isTo) {
    var html = '';
    if (isTo) {
        html = '<select class="nd-currencyDropdown" onchange="NetdaniaConverter.changeSelected(this.value,' + id + ');" name="to' + id + '" id="to' + id + '">';
    }
    else {

        html = '<select class="nd-currencyDropdown" onchange="NetdaniaConverter.changeToSelect(this.value,' + id + ');" name="from' + id + '" id="from' + id + '">';
    }
    var select = '';
    var selected = '';
    for (var currency in arr) {
        if (arr.hasOwnProperty(currency)) {
            if (arr[currency].n != undefined) {
                if (arr[currency].s == selectedValue) {
                    selected = ' selected="selected"';
                }
                select += '<option ' + selected + ' value="' + arr[currency].s + '" text="' + arr[currency].n + '"></option>';
                selected = '';
            }
        }
    }
    html += select;
    html += '</select>';
    return html;
}

NetdaniaConverter.Converter_rendResult = function (id) {
    var html = '<input readonly class="nd-input_amount" value="0,00" MAXLENGTH="8" id="result' + id + '"/>';
    return html;
}

NetdaniaConverter.Converter_rendRate = function (id, param) {
    var html = '<div class="nd-rateDiv"><span></span><span type="text" class="nd-result" id="rate' + id + param + '">...</span></span></div>';
    return html;
}

var Netdania = Netdania || {};
Netdania.Util = Netdania.Util || {};
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





