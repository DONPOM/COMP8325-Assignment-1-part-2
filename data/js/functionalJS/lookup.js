Netdania = Netdania || {};
Netdania.LookupComponent = Netdania.LookupComponent || {};

Netdania.LookupComponent.RenderLookupSearch = function (config) {
    var _providers = null;
    var _lookupSources = config.lookupSources;
    var _maxResults = config.maxResults || 6;
    var _containerId = config.containerId;
    var _callback = config.callback;
    var _theme = config.theme || '';
    var _connection = null;
    var _requestIds = [];
    var _data = [];
    var _isFirstReponse = true;
    var _currentProviders = [];
    var _executedSearch = [];
    var _sortedData = [];
    var _resultsWidth = 595;

    var _jqueryUICSSLink = '/cssgeneral/themes/Smoothness/jquery-ui.css';
    var _loadingImageLink = '/JSComponents/cssgeneral/images/loading.gif';
    var _searchImageLink = '/JSComponents/lookup/css/images/search.png';

    var _categories = { stock: 'STOCKS', forex: 'FOREX', index: 'INDICES', fund: 'FUNDS', bond: 'BONDS', future: 'FUTURES', option: 'OPTIONS' };

    this.initLookupSearch = function () {
        var host = Netdania.General.Config.Host; //the NetDania server to connect to

        var location = Netdania.Util.getLocation(Netdania.JsApi.Utilities.GetURL());
        var connectionParams = Netdania.General.Config.GetConnectionParams(location.pathname, Netdania.General.Config.URLConnectionParams);
        //console.log(connectionParams.pollingInterval, connectionParams.ct, 'lookup');
        var connectionConfig = {
            pollingInterval: connectionParams.pollingInterval, //milliseconds
            host: host,
            behavior: connectionParams.ct,//Netdania.JsApi.ConnectionType.LONGPOLLING, //LONG POLLING
            type: 1, //the server response format: JSONP
            v: 2
        }
        //create a json connection object
        _connection = new Netdania.JsApi.JSONConnection(connectionConfig);
        _connection.Connect();
         _providers = _getProviders(config.providers);
        _renderLookupSearch();
        _getLookupSources(config.lookupSources);
        //_setTheme();

    }

    var _getProviders = function (config) {
        var _providers = [];
        var _provider = {};
        for (var i = 0; i < config.length; i++) {
            var _item = config[i].split('|');
            _provider = {};
            _provider.provider = _item[0];
            _provider.market = _item[1];


            var _instrTypes = _item[2] ? _item[2].split(';') : [0, 1, 2, 3, 4, 5, 6, 7];
            var _instrTypesInt = [];
            for (var j = 0, len = _instrTypes.length; j < len; j += 1) {
                _instrTypesInt.push(+_instrTypes[j]);
            }
            _provider.instrTypes = _instrTypesInt;
            _provider.alias = _item[3];

            _providers.push(_provider);
        }
        return _providers;
    }

    var _getLookupSources = function (configSources) {
        var _sources = [];
        _categories = [];
        var _source = {};
        for (var i = 0; i < configSources.length; i++) {
            _source = {};
            _source.req_type = configSources[i].reqType;
            _source.directory = configSources[i].directory;
            _categories.push({ name: configSources[i].label, priority: i });
            //_sources.push(_provider);
        }
        return _sources;
    }

   /* _providers = _getProviders(config.providers);*/

    var _getProviderByProviderName = function (providers, provider) {
        for (var i = 0; i < providers.length; i++) {
            if (providers[i].provider === provider.name && providers[i].market === provider.market)
                return providers[i];
        }
        return '';
    }

    var _getCategory2 = function (type) {
        var _category = _categories.forex;

        switch (type) {
            case 0:
                _category = _categories.forex;
                break;
            case 4:
                _category = _categories.index;
                break;
            case 1:
                _category = _categories.stock;
                break;
            case 2:
            case 5:
            case 6:
                _category = _categories.bond;
                break;
            case 3:
                _category = _categories.future;
                break;
            //            case 5:                                                             
            //                _category = 'OPTION';                                                             
            //                break;                                                             
            //            case 6:                                                             
            //                _category = 'WARRANT';                                                             
            //                break;                                                                 
            case 7:
                _category = _categories.fund;
                break;
        }

        return _category;
    }

    var _getCategory = function (type) {
        var _category = _categories.forex;

        switch (type) {
            case 0:
                _category = _categories.forex;
                break;
            case 4:
                _category = _categories.index;
                break;
            case 1:
                _category = _categories.stock;
                break;
            case 2:
            case 5:
            case 6:
                _category = _categories.bond;
                break;
            case 3:
                _category = _categories.future;
                break;
            //            case 5:                                                              
            //                _category = 'OPTION';                                                              
            //                break;                                                              
            //            case 6:                                                              
            //                _category = 'WARRANT';                                                              
            //                break;                                                            
            case 7:
                _category = _categories.fund;
                break;
        }

        return _category;
    }


    var _getInstrumentType = function (type) {
        var _type = '';

        switch (type) {
            case 0:
                _type = 'Currency';
                break;
            case 1:
                _type = 'Stock';
                break;
            case 2:
                _type = 'Bond';
                break;
            case 3:
                _type = 'Future';
                break;
            case 4:
                _type = 'Index';
                break;
            case 5:
                _type = 'Option';
                break;
            case 6:
                _type = 'Warrant';
                break;
            case 7:
                _type = 'Fund';
                break;
        }

        return _type;
    }

    var _getLookupValues = function (data, category, directory) {
        //{"isin":"DK01","ticker":"ALM","name":"AlmBrand","type":2,"delay":0,"mic":"XCSE","mname":"OMX NORDIC","mcountry":"DENMARK","mcity":"COPENHAGEN","pname":"ms","psymbol":"ALM.co","rank":1}
        var values = [];
        if (data != undefined) {
            for (var i = 0; i < data.length; i++) {
                var _jsonData = data[i].h;
                var _obj = $.parseJSON(_jsonData);
                //values.push({ label: data[i].n, value: data[i].s, provider: provider.name, category: _getCategory(data[i].t), providerAlias: _getProviderByProviderName(_providers, provider).alias, type: _getInstrumentType(data[i].t) });
                values.push({ label: _obj.name, value: _obj.ticker || '', provider: _obj.pname, category: category/*_getCategory(_obj.type)*/, providerAlias: _obj.pname/*_getProviderByProviderName(_providers, _obj.pname).alias*/,
                    type: /*_getInstrumentType(_obj.type) +*/(_obj.mname.indexOf(_obj.mcity) == -1 ? _obj.mcity + " - " : '') + _obj.mname, delay: (_obj.delay === 0 ? 'Realtime' : (_obj.delay === -1 ? 'EOD' : _obj.delay + ' min. delayed')),
                    isin: _obj.isin || '', symbol: _obj.psymbol, priority: _getCategoryPriority(category), country: _obj.mcountry, directory: directory
                });
            }
        }

        return values;
    }

    function _arrayContains(arr, val, equals) {
        var i = arr.length;
        while (i--) {
            if (equals(arr[i], val)) {
                return true;
            }
        }
        return false;
    }

    var _removeDuplicates = function (arr, equals) {
        var originalArr = arr.slice(0);
        var i, len, j, val;
        arr.length = 0;

        for (i = 0, len = originalArr.length; i < len; ++i) {
            val = originalArr[i];
            if (!_arrayContains(arr, val, equals)) {
                arr.push(val);
            }
        }

        return arr;
    }

    var _getCategoryPriority1 = function (category) {
        switch (category) {
            case _categories.forex:
                return 1;
                break;
            case _categories.stock:
                return 2;
                break;
            case _categories.index:
                return 3;
                break;
            case _categories.bond:
                return 6;
                break;
            case _categories.fund:
                return 5;
                break;
            case _categories.future:
                return 4;
                break;
            default:
                return 1000;
                break;
        }
    }

    var _getCategoryPriority = function (category) {
        var _priority = 1000;

        for (var i = 0; i < _categories.length; i++) {
            if (_categories[i].name === category)
                _priority = _categories[i].priority + 1;
        }
        return _priority;
    }

    var _generateLookupSource = function (config) {
        var _source = {};
        //'{"req_type":0,"directory":"test","search":"' + request.term + '","markets":"[ALDP, ALXB, AMTS, APXL]","providers":"[idc_dla, idc_rtb]","index":15,"count":30}'
        _source.req_type = config.req_type;
        _source.directory = config.directory;
        _source.search = config.search;
        _source.markets = config.markets;
        _source.providers = config.providers;
        _source.index = config.index;
        _source.count = config.count;

        return JSON.stringify(_source);
    }

    var _setTheme = function () {
        //$.cookie('jquery-ui-theme', _theme == "white" ? 'Smoothness' : 'UI%20darkness');

        //        $('#' + _containerId).themeswitcher({ height: 0, initialText: '',
        //            loadTheme: 'UI darkness', buttonPreText: 'Skin: ', buttonHeight: 13
        //        });

        $('head').append(' <link rel="stylesheet" href="' + _jqueryUICSSLink + '" type="text/css" />');
    }

    function _isObjectEqual(object1, object2) {
        return object1.label === object2.label
        && object1.value === object2.value;
    }

    var _toTitleCase = function (str) {
        return str.replace(/\w\S*/g, function (txt) { return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase(); });
    }

    //    (function ($) {

    //        $(".ui-autocomplete-input").live("autocompleteopen", function () {
    //            var autocomplete = $(this).data("autocomplete"),
    //		menu = autocomplete.menu;

    //            if (!autocomplete.options.selectFirst) {
    //                return;
    //            }

    //            menu.activate($.Event({ type: "mouseenter" }), menu.element.children().first());
    //        });

    //    } (jQuery));


    var _compareCategories = function (a, b) {
        if (_getCategoryPriority(a.category) < _getCategoryPriority(b.category))
            return -1;
        if (_getCategoryPriority(a.category) > _getCategoryPriority(b.category))
            return 1;
        return 0;
    }

    //objs.sort(compare);

    var _renderLookupSearch = function () {

        var _searchHtml = '<div style="float:left">' +
                          '<input style="float:left; margin-top:0" value="&nbsp;\'EUR/USD\' or \'GOOG\'" class="ui-autocomplete-input ui-widget ui-widget-content ui-corner-all nd-lookup" id="lookup' + _containerId + '" /></div>' +
                            //'<button id="nd-lookup-icon' + _containerId + '" style="float:left"></button>'+
                            '<img src="' + _searchImageLink + '" style="position: absolute;top: 17px; left: 515px;"/>' +
                          '<div id="loading' + _containerId + '" style="display:none; float:left"><img src="' + _loadingImageLink + '" style="width:30px; height:26px"/>' +
                          '</div>';
        $('#' + _containerId).html(_searchHtml);

        $('#lookup' + _containerId).click(function () { this.value = ''; });

        $("#nd-lookup-icon" + _containerId).button({
            icons: {
                primary: "ui-icon-search"
            },
            text: false
        }).css('height', '24px').css('width', '28px').css('margin-top', '0px')
        .click(function () { return false; });

        //        $("#lookup" + _containerId).keypress(function (e) {

        //            var code = (e.keyCode ? e.keyCode : e.which);
        //            if (code == 13) {
        //                console.log(_data[0]);
        //                _callback(_data[0]);
        //            }

        //        });


        //        $("#lookup" + _containerId).result(function (event, data, formatted) {
        //            alert(data);
        //        });
        var _comboSelectedItem = null;

        var lookup = $("#lookup" + _containerId)
            .bind('keyup', function (e) {
                var code = (e.keyCode ? e.keyCode : e.which);
                if (code == $.ui.keyCode.ENTER) {
                    $(this).catcomplete("close");
                    if (_comboSelectedItem === null) {
                        _callback(_sortedData[0]);
                    }
                    else {
                        _callback(_comboSelectedItem);
                    }
                }
            })
            .catcomplete({
                appendTo: '#' + _containerId,
                source: function (request, response) {
                    if (request && request.term != " 'EUR/USD' or 'GOOG'") {

                        //                        response($.map([{ label: 'Searching...', category: '', value: '', provider: '', providerAlias: '', type: '', delay: '', isin: '', symbol: '', country: '', directory: ''}], function (item) {
                        //                            return {
                        //                                label: item.label,
                        //                                value: item.value,
                        //                                provider: item.provider,
                        //                                option: item,
                        //                                category: item.category,
                        //                                providerAlias: item.providerAlias,
                        //                                type: item.type,
                        //                                delay: item.delay,
                        //                                isin: item.isin,
                        //                                symbol: item.symbol,
                        //                                country: item.country,
                        //                                directory: item.directory
                        //                            }
                        //                        }));



                        $('#loading' + _containerId).css('display', 'block');
                        _requestIds = [];
                        _data = [];
                        _prioritiesArray = [];
                        _currentProviders = [];

                        if (_executedSearch[request.term]) {
                            _sortedData = _executedSearch[request.term];
                            response($.map(_executedSearch[request.term], function (item) {
                                return {
                                    label: item.label,
                                    value: item.value,
                                    provider: item.provider,
                                    option: item,
                                    category: item.category,
                                    providerAlias: item.providerAlias,
                                    type: item.type,
                                    delay: item.delay,
                                    isin: item.isin,
                                    symbol: item.symbol,
                                    country: item.country,
                                    directory: item.directory
                                }
                            }));
                            $('#loading' + _containerId).css('display', 'none');
                            return;
                        }

                        //                                                for (var i = 0; i < _providers.length; i++) {
                        //                                                    //contains
                        //                                                    var _req = Netdania.JsApi.Request.getReqObjInstrumentLookup(_providers[i].market, 0, request.term, 2, _providers[i].instrTypes, _maxResults / 2, _providers[i].provider);
                        //                                                    //starts with
                        //                                                    var _reqStWith = Netdania.JsApi.Request.getReqObjInstrumentLookup(_providers[i].market, 0, request.term, 1, _providers[i].instrTypes, _maxResults / 2, _providers[i].provider);
                        //                                                    _connection.AddRequest(_req);
                        //                                                    _connection.AddRequest(_reqStWith);

                        //                                                    _requestIds.push(_req.i);
                        //                                                    _requestIds.push(_reqStWith.i);
                        //                                                    _currentProviders[_req.i] = { name: _providers[i].provider, market: _providers[i].market, searchTerm: request.term };
                        //                                                    _currentProviders[_reqStWith.i] = { name: _providers[i].provider, market: _providers[i].market, searchTerm: request.term };
                        //                                                }


                        var _objStartDate = new Date();
                        //_objStartDate.setDate(_objStartDate.getDate() - 10);
                        //var _objEndDate = new Date();

                        for (var i = 0; i < _lookupSources.length; i++) {

                            var _lookupSourceConfig = _lookupSources[i]; //config.lookupSource;
                            _lookupSourceConfig.search = request.term;
                            var _lookupSource = _generateLookupSource(_lookupSourceConfig);
                            //'{"req_type":0,"directory":"test","search":"' + request.term + '","markets":"[ALDP, ALXB, AMTS, APXL]","providers":"[idc_dla, idc_rtb]","index":15,"count":30}';
                            var _req = Netdania.JsApi.Request.getReqObjNewsSearch(_lookupSource, "", 15, Math.round(_objStartDate.getTime() / 1000), Math.round(_objStartDate.getTime() / 1000), 0, "instrumentsdirectory");
                            _requestIds.push(_req.i);
                            _currentProviders[_req.i] = { name: _lookupSourceConfig.providers, market: _lookupSourceConfig.markets, searchTerm: request.term, label: _lookupSourceConfig.label, directory: _lookupSourceConfig.directory };
                            _connection.AddRequest(_req);
                        }

                        _connection.Flush();

                        if (_isFirstReponse) {
                            _isFirstReponse = false;
                            //_connection.addListener(Netdania.JsApi.Events.ONLOOKUP, function (lookupResponse) {
                            _connection.addListener(Netdania.JsApi.Events.ONNEWSSEARCH, function (lookupResponse) {

                                if ($.inArray(lookupResponse.id, _requestIds) === -1) {
                                    return;
                                }

                                var _responseData = _getLookupValues(lookupResponse.data, _currentProviders[lookupResponse.id].label, _currentProviders[lookupResponse.id].directory); //_lookupSources[lookupResponse.id]);

                                var _responseDataLimit = _responseData.slice(0, config.maxResults);
                                if (_responseDataLimit.length > 0) {
                                    _prioritiesArray[_responseDataLimit[0].priority + 'p'] = _responseDataLimit;
                                }
                                _data = [];
                                for (var p = 0; p <= _lookupSources.length; p++) {
                                    if (_prioritiesArray[p + 'p'] != null) {
                                        _data = _data.concat(_prioritiesArray[p + 'p']);
                                    }
                                }


                                //                                if (_data.length > 0 && (_data[0].priority < _responseDataLimit[0].priority)) {
                                //                                    _data = _data.concat(_responseDataLimit);
                                //                                }
                                //                                else {
                                //                                    _data = _responseDataLimit.concat(_data);
                                //                                }


                                _requestIds = $.grep(_requestIds, function (value) {
                                    return value != lookupResponse.id;
                                });



                                _sortedData = jQuery.ui.autocomplete.filter(_data, '');



                                if (_sortedData.length === 0 && _requestIds.length === 0) {
                                    _sortedData.push({ label: 'No item found!', category: '', value: '', provider: '', providerAlias: '', type: '', delay: '', isin: '', symbol: '', country: '', directory: '' });
                                }

                                //_sortedData = _removeDuplicates(_sortedData, _isObjectEqual);

                                //                                _sortedData = _sortedData.sort(function (a, b) {

                                //                                    var _aL = (a.label).toLowerCase().indexOf(_currentProviders[lookupResponse.id].searchTerm.toLowerCase());
                                //                                    var _bL = (b.label).toLowerCase().indexOf(_currentProviders[lookupResponse.id].searchTerm.toLowerCase());

                                //                                    var _categoryA = _getCategoryPriority(a.category);
                                //                                    var _categoryB = _getCategoryPriority(b.category);

                                //                                    return (_aL === -1 ? 1000 : _aL) - (_bL === -1 ? 1000 : _bL) + (_categoryA - _categoryB);
                                //                                });


                                //                                _sortedData = _sortedData.sort(function (a, b) {

                                //                                    var _categoryA = _getCategoryPriority(a.category);
                                //                                    var _categoryB = _getCategoryPriority(b.category);

                                //                                    return (_categoryA - _categoryB);

                                //                                });


                                //                                _sortedData = _sortedData.sort(function (a, b) {

                                //                                    var _categoryA = (a.priority);
                                //                                    var _categoryB = (b.priority);

                                //                                    return (_categoryA - _categoryB);

                                //                                });

                                //console.log(_sortedData);
                                //_sortedData = _sortedData.sort(_compareCategories);

                                if (_requestIds.length === 0) {
                                    $('#loading' + _containerId).css('display', 'none');
                                    _executedSearch[_currentProviders[lookupResponse.id].searchTerm + 'X'] = _sortedData;
                                }

                                response($.map(_sortedData, function (item) {
                                    return {
                                        label: item.label,
                                        value: item.value,
                                        provider: item.provider,
                                        option: item,
                                        category: item.category,
                                        providerAlias: item.providerAlias,
                                        type: item.type,
                                        delay: item.delay,
                                        isin: item.isin,
                                        symbol: item.symbol,
                                        country: item.country,
                                        directory: item.directory
                                    }
                                }));

                            });
                        }
                    }
                },
                minLength: 1,
                select: function (event, ui) {
                    if (ui.item != undefined) {
                        _callback(ui.item);
                        $("#lookup" + _containerId)[0].value = ui.item.label;
                    }
                    return false;
                },
                open: function () {
                    $(this).removeClass("ui-corner-all").addClass("ui-corner-top");
                    var position = $('#lookup-ul').position();
                    $('#' + _containerId + ' .ui-menu').width(_resultsWidth);
                    $('#lookup-ul').bgiframe1({ width: _resultsWidth, height: $('#lookup-ul').height(), left: position.left, top: position.top });
                    
                },
                close: function () {
                    $(this).removeClass("ui-corner-top").addClass("ui-corner-all");
                    $('.bgiframe').remove();
                },
                focus: function (event, ui) {
                    if (ui.item == undefined) {
                    } else {
                        $("#lookup" + _containerId)[0].value = ui.item.label;
                        _comboSelectedItem = ui.item;
                    }
                    event.preventDefault();
                },
                //autoFocus: true,
                delay: 300
            });
        //            .result(function (event, data, formatted) {
        //                alert(data);
        //            });


    }

    $.widget("custom.catcomplete", $.ui.autocomplete, {
        _renderMenu: function (ul, items) {


            var self = this,
            categories = {};

            ul.attr('id', 'lookup-ul');

            /* Build a dictionary/hash where keys are categories and values are 
            * arrays of items with that category 
            */
            $.each(items, function (index, item) {
                if (!categories.hasOwnProperty(item.category)) {
                    categories[item.category] = [item];
                } else {
                    categories[item.category] = categories[item.category].concat([item]);
                }
            });

            /* Iterate over the hash we just built and display a category followed by its
            * items.
            */

            //            categoriesArr = Object.keys(categories).sort(function (a, b) {
            //                if (_getCategoryPriority(a) < _getCategoryPriority(b)) return -1;
            //                if (_getCategoryPriority(a) > _getCategoryPriority(b)) return 1;
            //                return 0;
            //            });


            $.each(categories, function (category, items) {
                if (category) {
                    ul.append("<li class='ui-autocomplete-category'><span class='nd-lookup-category'>" + category + " (" + items.length + ")</span></li>");
                }
                $.each(items, function (index, item) {
                    self._renderItem(ul, item);
                });
            });
        },

        /* overwrite. make the matching string bold */
        _renderItem: function (ul, item) {
            var label = item.label.length <= 32 ? item.label.replace(new RegExp(
                "(?![^&;]+;)(?!<[^<>]*)(" + $.ui.autocomplete.escapeRegex(this.term) +
                ")(?![^<>]*>)(?![^&;]+;)", "gi"), "<span class='matchingClass'>$1</span>") :
                (item.label.substring(0, 30) + "...").replace(new RegExp(
                "(?![^&;]+;)(?!<[^<>]*)(" + $.ui.autocomplete.escapeRegex(this.term) +
                ")(?![^<>]*>)(?![^&;]+;)", "gi"), "<span class='matchingClass'>$1</span>");

            var symbol = item.value.length <= 6 ? item.value.replace(new RegExp(
                "(?![^&;]+;)(?!<[^<>]*)(" + $.ui.autocomplete.escapeRegex(this.term) +
                ")(?![^<>]*>)(?![^&;]+;)", "gi"), "<span class='matchingClass'>$1</span>") :
                (item.value.substring(0, 5) + "...").replace(new RegExp(
                "(?![^&;]+;)(?!<[^<>]*)(" + $.ui.autocomplete.escapeRegex(this.term) +
                ")(?![^<>]*>)(?![^&;]+;)", "gi"), "<span class='matchingClass'>$1</span>");

            var exch = item.type.length < 26 ? item.type : item.type.substring(0, 24) + '...';

            return $("<li title='" + item.value + " - " + item.label + " - " + item.type + (item.category === "Forex" ? "" : ("; ISIN - " + item.isin)) + "'></li>")
                .data("item.autocomplete", item)
                .append("<a>" +
                            "<table width='100%' cellpadding='0' cellspacing='0'><tr style='height:8px'>" +
                             "<td style='width:80px; margin-left:10px; height:8px'>" + symbol + "</td>" +
                            "<td style='width:220px;word-wrap: break-word;padding-right:5px'>" + label + "</td>" +
            //"<td style='float:right;width:50px'>" + item.providerAlias + "</td>" +


            //"<td style='text-transform:capitalize;'>" + _toTitleCase(exch) + "</td>" +
                            "<td style='text-transform:capitalize;'>" + exch + "</td>" +
                            "<td style='width:95px;font-style:italic'>" + item.delay + "</td>" +
                            "</tr></table>" +
                        "</a>"
                            )
                .appendTo(ul);
        }
    });
};



