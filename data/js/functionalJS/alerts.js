Netdania = Netdania || {};
Netdania.Alert = Netdania.Alert || {};
Netdania.Alert.AlertComponent = Netdania.Alert.AlertComponent || {};
var globalAlertObject;

Netdania.Alert.AlertComponent.RenderAlertComponent = function (config) {
    globalAlertObject = this;

    var _theme = config.theme || '';
    var _provider = config.provider;
    var _symbol = config.symbol;
    var _decimals;
    var _filter = config.filter || undefined; //[{field:'fieldName', value:'fieldValue'}]
    var _oneday = 1000 * 24 * 60 * 60;
    var _self = this;
    var _selectedField;
    var _selectedOperator = '>';
    var _selectedEmailType;
    var _currentAlertID = 0;
    var _deleteReqID = 0;
    var _editReqID = 0;
    var _addReqID = 0;
    var _lastReqID = 0;
    var _selectedAlertObject;
    var _eventOnProgress = false;
    var _currentQuoteList;
    var _pushDevices = [];

    var _alertsGridContainer = config.alertsGridContainer;
    var _alertsSentGridContainer = config.alertsSentGridContainer;
    var _alertsDeletedGridContainer = config.alertsDeletedGridContainer;

    var _alertsSentStartDateInput = config.alertsSentStartDateInput;
    var _alertsSentEndDateInput = config.alertsSentEndDateInput;
    var _alertsDeletedStartDateInput = config.alertsDeletedStartDateInput;
    var _alertsDeletedEndDateInput = config.alertsDeletedEndDateInput;

    var _alertsSentSearch = config.alertsSentSearch;
    var _alertsDeletedSearch = config.alertsDeletedSearch;

    var _getExpireDate = function () {

        var expireDate = $('#' + _expDate).datepicker("getDate");
        if (expireDate != undefined) {
            return Math.round(Math.abs(new Date() - new Date(expireDate) -1) / _oneday);
        }
        else {
            var returnedDate = new Date();
            return returnedDate.setDate(returnedDate.getDate + 365);
        }
    };


    var RESTRICTED_OPERATORS = [
        { t: '>', v: '>' },
        { t: '>=', v: '>=' },
        { t: '<', v: '<' },
        { t: '<=', v: '<=' }
    ];
    var CONTACT_TYPE = [
        { t: 'email', v: 'email' },
        { t: 'device', v: 'device' }
    ];

    Events.enable.call(this);
    var _alertsList = [];
    var _alertsSentList = [];
    var _alertsDeletedList = [];

    var _alertsLists = [
        [],
        []
    ];
    var AlertListType = { "ACTIVE": 0, "SENT": 1, "DELETED": 2 };

    config.fieldsControlName = 'nd-ddBidAsk';
    config.operatorsControlName = 'nd-ddOperator';
    config.valueAlertName = 'nd-txtValueAlert';
    config.emailInputName = 'nd-txtEmailAddress';
    config.contactTypeName = 'nd-ddContactType';
    config.messageAlertName = 'nd-txtMessageAlert';
    var _expDate = "nd-expireDatePicker_generalDivID";
    var _creationDate = "nd-creationDateHF";
    var _contactsList = [];
    var _connection;

    var _currentAddedAlert;
    //region properties
    var _getField = function () {
        return _selectedField;
    }

    var _setField = function (field) {
        _selectedField = field;
    }

    var _getOperator = function () {
        return _selectedOperator;
    }
    var _setOperator = function (operator) {
        _selectedOperator = operator;
    }

    var _getContactType = function () {
        return _selectedEmailType;
    }
    var _setContactType = function (emailType) {
        _selectedEmailType = emailType;
    }
    //end region properties

    this.clearFilter = function () {
        _filter = undefined;
    }

    this.setFilter = function (filter) {
        _filter = filter;
    }

    var _getContactsList = function (){
        return Netdania.Alert.AlertStorage.ContactsList;
    }
    var _setContactsList = function (val){
        Netdania.Alert.AlertStorage.ContactsList = val;
    }


    var AlertDeleteReason = (function () {
        var _DELETED_BY_USER = {code: 1, text : 'Deleted by user'};
        var _EXPIRED = {code: 2, text : 'Expired'};
        var _DELETED_BY_ADMIN = {code: 3, text : 'Deleted by admin'};

        return {
            Name: function (code) {
                switch (code)
                {
                    case _DELETED_BY_USER.code:
                        return _DELETED_BY_USER.text;
                    case _EXPIRED.code:
                        return _EXPIRED.text;
                    case _DELETED_BY_ADMIN.code:
                        return _DELETED_BY_ADMIN.text;
                    default :
                        return _DELETED_BY_USER.text;
                }
            },
            DELETED_BY_USER : _DELETED_BY_USER.code,
            EXPIRED:  _EXPIRED.code,
            DELETED_BY_ADMIN: _DELETED_BY_ADMIN.code
        }
    })();

    this.initRenderAlertComponent = function () {

        var connectionConfig = {
            pollingInterval: Netdania.JsApi.PoolingInterval.AUTO, //milliseconds
            host: Netdania.Alert.Config.HostAlerts,
            behavior: Netdania.JsApi.ConnectionType.AUTO, //LONG POLLING
            type: 1 //the server response format: JSONP
        }
        //create a json connection object
        _connection = new Netdania.JsApi.JSONConnection(connectionConfig);
        _connection.Connect();
        _getPushDevices();
        _renderAlertComponent();

        $(function () {
            $("#" + _expDate).datepicker({
                showOn: "both",
                buttonImage: "/Alerts/js/images/calendar.png",
                buttonImageOnly: true,
                minDate: 0,
                maxDate: "+365D",
                dateFormat: config.dateFormat,
                showOptions: { direction: "left" }
            }).next(".ui-datepicker-trigger").addClass("datePickerImage");

            var defaultDate = new Date();
            defaultDate.setMonth(defaultDate.getMonth() + 12);
            $('#' + _expDate).datepicker("setDate", defaultDate);

            $("#ui-datepicker-div").css('display', 'none');

            $("#" + _alertsSentStartDateInput).datepicker({
                showOn: "both",
                buttonImage: "/Alerts/js/images/calendar.png",
                buttonImageOnly: true,
                minDate: "-1y",
                maxDate: 0,
                dateFormat: config.dateFormat,
                showOptions: { direction: "right" }
            }).next(".ui-datepicker-trigger").addClass("datePickerImage");

            var defaultDate = new Date();
            defaultDate.setTime(defaultDate.getTime() - (30 * _oneday));
            $('#' + _alertsSentStartDateInput).datepicker("setDate", defaultDate);

            //region Sent End Date Initialization
            $("#" + _alertsSentEndDateInput).datepicker({
                showOn: "both",
                buttonImage: "/Alerts/js/images/calendar.png",
                buttonImageOnly: true,
                minDate: "-1y",
                maxDate: 0,
                dateFormat: config.dateFormat,
                showOptions: { direction: "left" }
            }).next(".ui-datepicker-trigger").addClass("datePickerImage");

            var defaultDate = new Date();
            defaultDate.setTime(defaultDate.getTime());
            $('#' + _alertsSentEndDateInput).datepicker("setDate", defaultDate);
            //endregion

            $("#" + _alertsSentSearch).click(function () {
                _self._showLoader();
                _self._refreshSentAlertsGrid();
            });

            $("#" + _alertsDeletedStartDateInput).datepicker({
                showOn: "both",
                buttonImage: "/Alerts/js/images/calendar.png",
                buttonImageOnly: true,
                minDate: "-1y",
                maxDate: 0,
                dateFormat: config.dateFormat,
                showOptions: { direction: "right" }
            }).next(".ui-datepicker-trigger").addClass("datePickerImage");

            var defaultDate = new Date();
            defaultDate.setTime(defaultDate.getTime() - (30 * _oneday));
            $('#' + _alertsDeletedStartDateInput).datepicker("setDate", defaultDate);

            //region Sent End Date Initialization
            $("#" + _alertsDeletedEndDateInput).datepicker({
                showOn: "both",
                buttonImage: "/Alerts/js/images/calendar.png",
                buttonImageOnly: true,
                minDate: "-1y",
                maxDate: 0,
                dateFormat: config.dateFormat,
                showOptions: { direction: "left" }
            }).next(".ui-datepicker-trigger").addClass("datePickerImage");

            var defaultDate = new Date();
            defaultDate.setTime(defaultDate.getTime());
            $('#' + _alertsDeletedEndDateInput).datepicker("setDate", defaultDate);
            //endregion

            $("#" + _alertsDeletedSearch).click(function () {
                _self._showLoader();
                _self._refreshDeletedAlertsGrid();
            });
        });
    }

    this.updatePushDevices = function(){
        _getPushDevices();
    }

    var _getPushDevices = function () {

        /*
        Get the push devices from session storage
         */
        _pushDevices = Netdania.Alert.AlertStorage.GetAllPushDevices();

        /*
        Set default contacts
         */
        defaultPushDevices();


        $("#contacts-div").html("");
        $("#contacts-div").html(RenderContactsList());
        _setDeleteEvents();
    }

    this.initAddAlertForm = function (settings) {
        _provider = config.provider = settings.provider || _provider;
        _symbol = config.symbol = settings.symbol || _symbol;
        _decimals = settings.decimals;
        _contactsList = defaultContactsList();
        _setContactsList(_contactsList);
        _currentAlertID = 0;
        clearAlertForm();
        _currentQuoteList = Netdania.Alert.QuoteList({ operatorID: config.operatorsControlName, initialUpdate: true, updateField: 'nd-txtValueAlert', symbol: settings.symbol, provider: settings.provider, containerId: 'nd-instrumentBidAsk', selectedField: (alert && alert.Instrument) ? alert.Instrument.getField() : '10', callbackFunction: _self.updateAllowedFields,  ddHtmlElementID: "nd-alert-dd", decimals: _decimals });
        $("#" + config.emailInputName).keyup(validateEmail).focus(validateEmail);

        $("#" + config.operatorsControlName).change(function () {
            validateAlertValue(false);
        });

        $("#" + config.valueAlertName).keyup(function () {
            validateAlertValue(false);
        }).focus(function () {
                validateAlertValue(false);
            }).focusout(function () {
                validateAlertValue(false);
            });


        $("#contacts-div").html("");
        $("#contacts-div").html(RenderContactsList());
        _setDeleteEvents();

    }

    var _renderAlertComponent = function () {

        _renderAddAlertDialog();

        $("#nd-btnAddAlert").button().bind('click touchstart', function (event) {
            try {
                _self._showLoader();

                config.relationalOperator = _getOperator();
                var txtValueAlert = $("#" + config.valueAlertName);
                if (txtValueAlert != undefined && txtValueAlert.length > 0) {
                    config.valueAlert = txtValueAlert.autoNumeric('get');
                }
                else {
                    config.valueAlert = 1;
                }

                config.creationDate =  $("#" + _creationDate).val();

                var messageAlert = $("#" + config.messageAlertName);
                if (messageAlert != undefined && messageAlert.length > 0) {
                    config.alertName = messageAlert[0].value;
                }
                config.alertFields = $("#" + config.fieldsControlName).val();
                _setField({ v: $("#" + config.fieldsControlName).val(), f: $("#" + config.fieldsControlName).html() });

                config.daysToLive = getExpireDate();

                if (validateAlertValue(true)) {
                    _addAlertOnCallback();
                }
                else {
                    _self._hideLoader();
                }
            }
            catch (err) {
                _self._hideLoader();
                Netdania.JsApi.Utilities.LogError("Alerts._renderAlertComponent", err);
            }
        });
    }

    var _addAlertOnCallback = function () {
        var alertExist = false;
        var alertDeleted = true;
        var storageAlerts = Netdania.Alert.AlertStorage.GetAllAlerts();
        if (storageAlerts != undefined) {
            for (var i = 0; i < storageAlerts.length; i++) {
                if (storageAlerts[i].symbol == config.symbol && storageAlerts[i].operator == config.relationalOperator.v && storageAlerts[i].alertValue == config.valueAlert) {
                    alertExist = true;
                    break;
                }

                if (_currentAlertID > 0 && _currentAlertID == storageAlerts[i].id) {
                    alertDeleted = false;
                    break;
                }

            }
        }

        if (_currentAlertID > 0 && alertDeleted) {
            $("#nd-alert_dialog-form").dialog("close");
            _self._hideLoader();
            customPopup = new Netdania.Util.PopUp();
            customPopup.initPopup({ containerName: 'popupContainer', cssClass: "alert-popup" });
            customPopup.showPopup({ title: 'Edit alert', message: 'The alert no longer exist!', content: '', messageType: 'warning', noCancelButton: true,
                callbackFunction: function () {
                    customPopup.closePopup();
                }
            });
        }
        else
        if (!alertExist || _currentAlertID > 0) {
            _addAlert(config);
        }
        else {
            if (alertExist) {
                _self._hideLoader();
                customPopup = new Netdania.Util.PopUp();
                customPopup.initPopup(configPopup);
                customPopup.showPopup({ title: 'Edited alert', message: 'An alert with condition ' + config.relationalOperator.v + config.valueAlert + ' is already added!', content: '', messageType: 'successfull', noCancelButton: true,
                    callbackFunction: function () {
                        customPopup.closePopup();
                    }
                });
            }
        }
    }

    var alertsExists = function (alertObjToCheck) {
        var alertExist = false;
        for (var i = 0; i < _alertsList.length; i++) {
            var alertObject = _alertsList[i].getAlertObject();
            if (alertObject.Instrument.getSymbol() == alertObjToCheck.getInstrument().getSymbol() && alertObject.Condition.getRelationalOperator() == alertObjToCheck.getCondition().getRelationalOperator().v && alertObject.Condition.getAlertValue() == alertObjToCheck.getCondition().getAlertValue()) {
                alertExist = true;
                break;
            }
        }
        return alertExist;
    }
    var _getAlertByID = function (alertID) {
        var alertUndefined;
        for (var i = 0; i < _alertsList.length; i++) {
            var alertObject = _alertsList[i].getAlertObject();
            if (alertObject.ALERT_ID == alertID) {
                return alertObject
            }
        }
        return alertUndefined;
    }

    this._addAlertActive = function (alertObject) {

        for (var i = 0; i < _alertsList.length; i++) {
            var alertObj = _alertsList[i].getAlertObject();
            if (alertObj.ALERT_ID == alertObject.getAlertObject().ALERT_ID) {
                _alertsList[i] = alertObject;
                return;
            }
        }
        _alertsList.push(alertObject);
    }

    this._addAlertToList = function (alertObject, AlertType) {

        for (var i = 0; i < _alertsLists[AlertType].length; i++) {
            var alertObj = _alertsLists[AlertType][i].getAlertObject();
            if (alertObj.ALERT_ID == alertObject.getAlertObject().ALERT_ID) {
                _alertsLists[AlertType][i] = alertObject;
                return;
            }
            _alertsLists[AlertType].push(alertObject);
        }
    }


    this._replaceAlert = function (alertObject) {

        for (var i = 0; i < _alertsList.length; i++) {
            var alertObj = _alertsList[i].getAlertObject();
            if (alertObj.ALERT_ID == alertObject.getAlertObject().ALERT_ID) {
                _alertsList[i] = alertObject;
                return;
            }
        }
    }

    this._deleteAlert = function (alertID) {

        for (var i = 0; i < _alertsList.length; i++) {
            var alertObj = _alertsList[i].getAlertObject();
            if (alertObj.ALERT_ID == alertID) {
                _alertsList.splice(i, 1)
                return;
            }
        }
    }

    this.updateAllowedFields = function (instrumentRealType, selectedField) {
        switch (instrumentRealType) {
            case 0:
            {
                alertFields = [
                    { t: 'Bid', v: '10' },
                    { t: 'Ask', v: '11' }
                ];
                selectedField = "10";
                break;
            }
            case 1:
            {
                alertFields = [
                    { t: 'Last', v: '6' }
                ];
                selectedField = "6";
                break;
            }
            case 2:
            {
                alertFields = [
                    { t: 'Bid', v: '10' },
                    { t: 'Ask', v: '11' },
                    { t: 'Last', v: '6' }
                ];
                selectedField = "10";
                break;
            }
        }
        $('#nd-allowedAlertsFields').html(_renderSelectorControl(config.fieldsControlName, alertFields));
        $('#' + config.fieldsControlName).val(selectedField);
        _setField({ v: selectedField });
        $('#' + config.fieldsControlName).change(function (event) {
            var selectedObject;
            try {
                selectedObject = event.target.selectedOptions[0];
            }
            catch (err) {
            }
            if (!selectedObject) {
                selectedObject = event.target[event.target.selectedIndex];
                selectedObject.label = selectedObject.text;
            }
            _setField(selectedObject.value);
        });
        _self._hideLoader();
    }

    this.checkSymbol = function (symbProv) {
        var symbolAndProvider = symbProv.split('|');
        var provider = symbolAndProvider[1];
        var selectedSymbol = symbolAndProvider[0];
        var providerAllowed = false;
        var symbolAllowed = false;

        if (config.allowedAlertProviders.length <= 0) {
            providerAllowed = true;
            symbolAllowed = true;
        }
        else {
            for (var i = 0; i < config.allowedAlertProviders.length; i++) {
                if (config.allowedAlertProviders[i].p == provider) {
                    providerAllowed = true;

                    var instrumentsSymbol = '';
                    if (config.allowedAlertProviders[i].s) {
                        instrumentsSymbol = config.allowedAlertProviders[i].s.split('|');
                    }
                    //if no symbol is defined then, all symbols are allowed for alert
                    if (instrumentsSymbol.length == 0) {
                        symbolAllowed = true;
                        break;
                    }
                    for (var k = 0; k < instrumentsSymbol.length; k++) {
                        if (selectedSymbol == instrumentsSymbol[k]) {
                            symbolAllowed = true;
                            break;
                        }
                    }
                    break;
                }
            }
        }
        if (!symbolAllowed || !providerAllowed) {
            customPopup = new Netdania.Util.PopUp();
            customPopup.initPopup(configPopup);
            customPopup.showPopup({ title: 'Error', message: 'You are not allowed to add alert on this instrument!', showButtons: true, messageType: 'warning',
                callbackFunction: function () {
                    customPopup.closePopup();
                }
            });
            return false;
        }

        return true;
    }

    var validateAlertValue = function (validateProviderAndSymbol) {
        var valueToCheck = 0;
        var bidAsk = _getField().v;
        var provider = _getFieldValue('p');
        var selectedSymbol = _getFieldValue('s');
        var providerAllowed = false;
        var symbolAllowed = false;
        if (validateProviderAndSymbol) {
            if (config.allowedAlertProviders.length <= 0) {
                providerAllowed = true;
                symbolAllowed = true;
            }
            else {
                for (var i = 0; i < config.allowedAlertProviders.length; i++) {
                    if (config.allowedAlertProviders[i].p == provider) {
                        providerAllowed = true;
                        var instrumentsSymbol = '';
                        if (config.allowedAlertProviders[i].s) {
                            instrumentsSymbol = config.allowedAlertProviders[i].s.split('|');
                        }
                        //if no symbol is defined then, all symbols are allowed for alert
                        if (instrumentsSymbol.length == 0) {
                            symbolAllowed = true;
                            break;
                        }
                        for (var k = 0; k < instrumentsSymbol.length; k++) {
                            if (selectedSymbol == instrumentsSymbol[k]) {
                                symbolAllowed = true;
                                break;
                            }
                        }
                        break;
                    }
                }
            }
            if (!providerAllowed) {
                customPopup = new Netdania.Util.PopUp();
                customPopup.initPopup(configPopup);
                customPopup.showPopup({ title: 'Error', message: 'You are not allowed to add alert on this instrument!', showButtons: true, messageType: 'successfull', noCancelButton: true,
                    callbackFunction: function () {
                        customPopup.closePopup();
                    }
                });
                return false;
            }
            if (!symbolAllowed) {
                customPopup = new Netdania.Util.PopUp();
                customPopup.initPopup(configPopup);
                customPopup.showPopup({ title: 'Error', message: 'You are not allowed to add alert on this instrument!', showButtons: true, messageType: 'successfull', noCancelButton: true,
                    callbackFunction: function () {
                        customPopup.closePopup();
                    }
                });
                return false;
            }
        }
        valueToCheck = _getFieldValue(bidAsk);
        var valueAlert = $("#" + config.valueAlertName);
        var controlOperator = $("#" + config.operatorsControlName);

        var valueOperator = _getOperator().v;
        if (controlOperator != null && controlOperator !== undefined) {
            valueOperator = controlOperator.find(":selected").text();
        }

        $("#AlertError").html("");

        //var alertValueNumber = parseFloat(valueAlert.val().replace(',', '.').replace(' ', ''));
        var alertValueNumber = parseFloat(valueAlert.autoNumeric('get'));

        if (isNaN(alertValueNumber)) {
            valueAlert.addClass("invalidValueCssClass");
            $("#AlertError").html("Value is not a valid number.");
            return false;
        }
        switch (valueOperator) {
            case '>':
            {
                if (alertValueNumber < valueToCheck) {
                    valueAlert.addClass("invalidValueCssClass");
                    $("#AlertError").html("Value is smaller than the Field");
                    return false;
                }
                else {
                    valueAlert.removeClass("invalidValueCssClass");
                }
                break;
            }
            case '>=':
            {
                if (alertValueNumber <= valueToCheck) {
                    valueAlert.addClass("invalidValueCssClass");
                    $("#AlertError").html("Value is smaller or equal than the Field");
                    return false;
                }
                else {
                    valueAlert.removeClass("invalidValueCssClass");
                }
                break;
            }
            case '<':
            {
                if (alertValueNumber > valueToCheck) {
                    valueAlert.addClass("invalidValueCssClass");
                    $("#AlertError").html("Value is bigger than the Field");
                    return false;
                }
                else {
                    valueAlert.removeClass("invalidValueCssClass");
                }
                break;
            }
            case '<=':
            {
                if (alertValueNumber >= valueToCheck) {
                    valueAlert.addClass("invalidValueCssClass");
                    $("#AlertError").html("Value is bigger or equal than the Field");
                    return false;
                }
                else {
                    valueAlert.removeClass("invalidValueCssClass");
                }
                break;
            }
        }
        return true;
    }

    function validateEmail() {
        var email = $("#" + config.emailInputName);

        // var contactType = $("#" + config.contactTypeName).val();
        if (email.val() == '') {
            email.removeClass("invalidValueCssClass");
            return true;
        }
        if (!Netdania.Util.validateEmail(email.val())) {
            email.addClass("invalidValueCssClass");
            return false;
        }
        else {
            email.removeClass("invalidValueCssClass");
        }
        return true;
    }

    var _getFieldValue = function (bidAsk) {
        var selectedObject = _currentQuoteList.getSelectedObject();
        for (var i = 0; selectedObject && i < selectedObject.length; i++) {
            if (selectedObject[i].f.toString() === bidAsk) {
                return selectedObject[i].v;
            }
        }
        return '';
    }

    var _addAlert = function (config) {
        _setAllContacts(config);

        var alert = new Netdania.Alert.AlertWrapper(config);

        alert.getAlertObject().ALERT_ID = _currentAlertID;

        if (alert.getAlertObject().EMAIL_CONTENT !== undefined) {
            if (alert.getAlertObject().EMAIL_CONTENT.split("Message: ")[1] === undefined) {
                alert.getAlertObject().EMAIL_CONTENT += "\nMessage: " + config.alertName;
            }
        }
        var simpleCondition = new Netdania.Alert.SimpleConditionWrapper();
        simpleCondition.SimpleConditionWrapper();
        if (config.creationDate != "") {
            simpleCondition.setTimestamp(config.creationDate);
        }
        simpleCondition.setTimeout(config.daysToLive);
        simpleCondition.setRelationalOperator(config.relationalOperator);
        simpleCondition.setAlertValue(Netdania.Util.Round(config.valueAlert.replace(',', '.'), 4));

        alert.setCondition(simpleCondition);
        var instrument = new Netdania.Alert.AlertInstrument(_getFieldValue('s'), _getFieldValue('p'), Netdania.Util.Utf8.encode(_getFieldValue('25')), Netdania.Alert.Types.PRICE);
        if (_selectedAlertObject) {
            instrument.setSymbol(_selectedAlertObject.Instrument.getSymbol());
        }
        instrument.setField(config.alertFields);
        alert.setInstrument(instrument);
        alert.Instrument = instrument;
        alert.setConditionToString();
        alert.Condition = simpleCondition;

        alert.setEmailContent(instrument.getName(), config.valueAlert, instrument.getField());
        _currentAddedAlert = alert;
        var addAlertRequest;

        $("#nd-alert_dialog-form").dialog("close");
        _self._hideLoader();


        if (_currentAlertID > 0) {
            addAlertRequest = Netdania.JsApi.Request.getReqObjAlertEdit(alert.getAlertObject());
            _editReqID = addAlertRequest.i;
            _connection.AddRequest(addAlertRequest);
            _connection.Flush();

            _connection.addListener(Netdania.JsApi.Events.ONALERTEDIT, function (alertAddedResponse) {
                if (alertAddedResponse && _editReqID == alertAddedResponse.id) {
                    _connection.removeListener(Netdania.JsApi.Events.ONALERTEDIT);
                    _currentAlertID = 0;

                    /*
                     We need to remove the request (No other request will be made)
                     */
                    _connection.removeRequests([_editReqID]);
                    _connection.Flush();
                }
            });
            _connection.addListener(Netdania.JsApi.Events.ONERROR, function (alertErrorResponse) {
                if (alertErrorResponse && alertErrorResponse.type == Netdania.JsApi.Alert.Commands.ERROR && _editReqID == alertErrorResponse.id) {
                    _connection.removeListener(Netdania.JsApi.Events.ONALERTADDED);
                }
            });


        }
        else {
            if (!alertsExists(alert)) {

                addAlertRequest = Netdania.JsApi.Request.getReqObjAlertAddAlert(alert.getAlertObject());
                _addReqID = addAlertRequest.i;
                _connection.AddRequest(addAlertRequest);
                _connection.Flush();

                _connection.addListener(Netdania.JsApi.Events.ONALERTADDED, function (alertAddedResponse) {
                    if (alertAddedResponse && alertAddedResponse.type == Netdania.JsApi.Alert.Commands.ADD_ALERT && _addReqID == alertAddedResponse.id) {
                        _connection.removeListener(Netdania.JsApi.Events.ONALERTADDED);
                        _self._refreshGrid();
                        _currentAddedAlert = null;

                        /*
                         We need to remove the request (No other request will be made)
                         */
                        _connection.removeRequests([_addReqID]);
                        _connection.Flush();
                    }
                });
                _connection.addListener(Netdania.JsApi.Events.ONERROR, function (alertErrorResponse) {
                    if (alertErrorResponse && alertErrorResponse.type == Netdania.JsApi.Alert.Commands.ERROR && _addReqID == alertErrorResponse.id) {
                        _alertErrorCodeAlert(alertErrorResponse);
                        _connection.removeListener(Netdania.JsApi.Events.ONALERTADDED);
                    }
                });

            }

        }
    };

    var _alertErrorCodeAlert = function (alertErrorResponse) {
        try {
            if (alertErrorResponse != undefined && alertErrorResponse.data != undefined && alertErrorResponse.data.f != undefined) {
                for (var i = 0; i < alertErrorResponse.data.f.length; i++) {
                    if (alertErrorResponse.data.f[i].f == Netdania.JsApi.Alert.Fields.ERROR_CODE) {
                        if (alertErrorResponse.data.f[i].v != undefined && alertErrorResponse.data.f[i].v[0] != undefined) {
                            switch (alertErrorResponse.data.f[i].v[0]) {
                                case Netdania.JsApi.Alert.ErrorCodes.ADD_ALERT_MAX_ALERTS_EXCEEDED_ERROR:
                                {
                                    customPopup = new Netdania.Util.PopUp();
                                    customPopup.initPopup({ containerName: 'popupContainer', cssClass: "alert-popup" });
                                    customPopup.showPopup({ title: 'Error adding alert', message: 'The maximum alerts limit has been reached!', content: '', messageType: 'warning', noCancelButton: true,
                                        callbackFunction: function () {
                                            customPopup.closePopup();
                                        }
                                    });
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        catch
            (err) {
            Netdania.JsApi.Utilities.LogError("Alert._alertErrorCodeAlert", err);
        }

    }


    var _setAllContacts;
    _setAllContacts = function (config) {
        config.emailAddress = "";
        _contactsList = _getContactsList();
        for (var i = 0; i < _contactsList.length; i++) {
            switch (_contactsList[i].t) {
                case CONTACT_TYPE[0].t:
                {
                    if (config.emailAddress) {
                        config.emailAddress += ';' + _contactsList[i].v;
                    }
                    else {
                        config.emailAddress = _contactsList[i].v;
                    }
                    break;
                }
            }
        }

        config.pushDevices = "";
        for (var i = 0; i < _pushDevices.length; i++) {
            if (_pushDevices[i].selected) {
                if (config.pushDevices == "")
                    config.pushDevices += _pushDevices[i].id;
                else
                    config.pushDevices += ',' + _pushDevices[i].id;
            }
        }
    };

    var getExpireDate;
    getExpireDate = function () {

        var expireDate = $('#' + _expDate).datepicker("getDate");
        expireDate.setDate(expireDate.getDate());

        var startDate = new Date();
        startDate.setHours(0,0,0,0);
        var creationDate = $('#' + _creationDate).val();
        if (creationDate != "") {
            var startDate= new Date(creationDate * 1);
        }


        if (expireDate != undefined) {
            return Math.ceil(Math.abs(startDate - new Date(expireDate)) / _oneday);
        }
        else {
            var returnedDate = new Date();
            return returnedDate.setDate(returnedDate.getDate + 365);
        }
    };
    var _renderAddAlertDialog;
    _renderAddAlertDialog = function () {

        var table =
            '<div id="nd-instrumentBidAsk"></div>' +
            '<div id="AlertFieldsTable">' +
                    '<table border="0" width="100%" cellspacing="2" cellpadding="2" >' +
                    '<thead class="headCls">' +
                        '<tr>' +
                            '<th class="headCls" width="10%"><div class="alert-nameHeaderClass">Field</div></th>' +
                            '<th class="headCls" width="10%"><span></span></th>' +
                            '<th class="headCls" width="10%"><div class="alert-nameHeaderClass">Value</div></th>' +
                            '<th class="headCls" width="50%"><div class="alert-nameHeaderClass">Message</div></th>' +
                        '</tr>' +
                    '</thead>' +
                    '<tr class="nd-alert-add-form-condition" style="vertical-align:top;">' +
                        '<td  id="nd-allowedAlertsFields" class="styled-select"></td>' +
                        '<td class="styled-select">' + _renderSelectorControl(config.operatorsControlName, RESTRICTED_OPERATORS) + '</td>' +
                        '<td class="styled-select"><input type="text" id="nd-txtValueAlert"/></td>' +
                        '<td class="styled-select" ><input id="nd-txtMessageAlert" class="nd-alerts-messageAlert" maxlength="99"/></td>' +
                    '</tr>' +
                    '<tr>' +
                        '<td colspan="4"><div id="AlertError" style="color:red;"></div></td>' +
                    '</tr>' +
                    '<tr>' +
                        '<td colspan="2"></td>' +
                        '<td><div id="nd-alert-dd"></div></td>' +
                        '<td><div id="nd-input-date-field"><input type="hidden" id="'+_creationDate+'"><b>Expiry</b><div id="nd-input-date-container"><input type="text" readonly="true" class="nd-input-date" id="' + _expDate + '" /></div></div></td>' +
                    '</tr>' +
                    '</table>' +
            '</div>' +
            '<div id="contacts-popup">' +
            '<div id="contacts-div"></div>' +
            '</div>' +
            '<div id="footer-div">' +
                '<table cellpadding="0" cellspacing="0" width="100%">' +
                    '<tr><td>' +
                    '<table><tr><td>' +
                        '<input type="text" id="' + config.emailInputName + '" class="' + config.emailInputName + '" />' +
                            '</td>' +
                            '<td>' +
                                '<input type="button" id="nd-btnAddContact"  value="Add email"   />' +
                                '</td></tr></table>' +
                                '</td><td>' +
                                    '<input type="button"  id="nd-btnAddAlert"  value="Save alert" autofocus="true"/>' +
                '</td></tr></table>' +
            '</div>';




        _setOperator(RESTRICTED_OPERATORS[0]);
        _setContactType(CONTACT_TYPE[0]);




        $("#" + config.containerId).html(table);

        $("#" + config.valueAlertName).autoNumeric('init',{vMin:0, vMax:999999999.99999, mDec:5, aPad:false});
        $("#" + config.valueAlertName).change(function(event) {
            $("#" + config.valueAlertName).autoNumeric('update',{vMin:0, vMax:999999999.99999, mDec:5, aPad:false});
        });
        $("#contacts-div").html(RenderContactsList());
        _setDeleteEvents();

        $('#' + config.operatorsControlName).change(function (event) {
            var selectedObject;
            try {
                selectedObject = event.target.selectedOptions[0];
            }
            catch (err) {
            }

            if (!selectedObject) {
                selectedObject = event.target[event.target.selectedIndex];
                selectedObject.label = selectedObject.text;
            }
            _setOperator({ t: selectedObject.label, v: selectedObject.value });
        });
        $('#' + config.contactTypeName).change(function (event) {
            var selectedObject;
            try {
                selectedObject = event.target.selectedOptions[0];
            }
            catch (err) {
            }

            if (!selectedObject) {
                selectedObject = event.target[event.target.selectedIndex];
                selectedObject.label = selectedObject.text;
            }
            if (event.target.selectedIndex > CONTACT_TYPE.length - 1) {
                _setContactType({ t: selectedObject.attributes[1].value, v: selectedObject.label, id: selectedObject.value });
                $("#" + config.emailInputName).val(selectedObject.label);
                $("#" + config.emailInputName)[0].step = selectedObject.attributes[1].value;
            }
            else {
                _setContactType({ t: selectedObject.label, v: selectedObject.value });
                $("#" + config.emailInputName).val(_getFirstContactOfType(selectedObject.value));
            }
        });


        $("#nd-btnViewContacts")
            .button()
            .click(function () {
                $("#contacts-div").html(RenderContactsList());
                _setDeleteEvents();
            });
        $("#nd-btnAddContact")
            .button().bind('click', function () {
                var value = $("#" + config.emailInputName).val();
                var id;
                var contactType = CONTACT_TYPE[0].t;
                if (value != '') {
                    if (validateEmail()) {
                        var allreadyExists = false;
                        _contactsList = _getContactsList();
                        for (var i = 0; i < _contactsList.length; i++) {
                            if (_contactsList[i].v == value) {
                                allreadyExists = true;
                                break;
                            }
                        }

                        if (_contactsList.length <= 0 || !allreadyExists) {

                            pushContactToList([value], contactType);

                            Netdania.Alert.DefaultContactStorage.AddContact({v:value,t:contactType});
                            $("#" + config.emailInputName).val('');
                        }

                        $("#contacts-div").html(RenderContactsList());
                        _setDeleteEvents();
                    }
                }
                else {
                    $("#" + config.emailInputName).addClass("invalidValueCssClass");
                }
            });
    };

    //create the html conten that display the contacts list table 
    var RenderContactsList = function () {
        _contactsList = _getContactsList();
                var tbl = "<div><div class='nd-alert-recipients'>Recipients</div><div id='nd-alert-recipients-contacts'><table id='ContactListTable' width='100%'  cellspacing='0' cellpadding='0' >{0}</table></div></div>";
        var row = "<tr {3}><td class='nameColumnRecipients'>{1}</td><td>{2}</td></tr>{0}";
        var index = 0;
        for (var i = 0; i < _contactsList.length; i++) {
            tbl = tbl.replace("{0}", row.replace("{1}", _contactsList[i].v).replace("{2}", "<img id='nd-DeleteContactButton" + i + "' title='Delete contact' class='nd-deletecontactbutton' src='/JSComponents/Alerts/images/delete.png' contact=" + _contactsList[i].v + " /> ").replace("{3}", (index % 2 == 0) ? "class=gridDefaultContentClassRecipients" : "class=gridDefaultContentOddClassRecipients"));
            index++;
        }

        for (var i = 0; i < _pushDevices.length; i++) {
            tbl = tbl.replace("{0}", row.replace("{1}", _pushDevices[i].v + ' [ ' + _pushDevices[i].appType + ' ] ').replace("{2}", ((_pushDevices[i].selected) ? "<input type='checkbox' checked  class='nd-pushDevices-check' device=" + _pushDevices[i].id + " />" : "<input type='checkbox' class='nd-pushDevices-check'  device=" + _pushDevices[i].id + " />")).replace("{3}", (index % 2 == 0) ? "class=gridDefaultContentClassRecipients" : "class=gridDefaultContentOddClassRecipients"));
            index++;
        }


        return tbl.replace("{0}", "");
    }

    var _hideContactsList = function (eventObj) {
        if ($("#contacts-div") != undefined && $("#contacts-div").length > 0) {
            if ($("#contacts-div")[0].style.display == "block" && eventObj.target.id != "nd-btnViewContacts") {
                $("#nd-btnViewContacts").val('Show contacts');
                // $("#contacts-div").css('display','');
            }
        }
    }

    //return the first contact of the specified type from the contacts list
    var _getFirstContactOfType = function (contactType) {
        for (var i = 0; i < _contactsList.length; i++) {
            if (_contactsList[i].t == contactType) {
                return _contactsList[i].v;
            }
        }
        return '';
    }

    var _setDeleteEvents = function () {
        $(".nd-deletecontactbutton").bind('click', function (eventObject) {

            customPopup = new Netdania.Util.PopUp();
            customPopup.initPopup({ containerName: 'popupContainer', cssClass: "alert-popup" });
            customPopup.showPopup({ title: 'Delete confirmation', message: 'Are you sure you want to delete this contact?', showButtons: true, messageType: 'successfull',
                callbackFunction: function () {
                    $("#contacts-div").html("");
                    _contactsList = _getContactsList();
                    var deletedContact = $(eventObject.target).attr('contact');
                    for (var i = 0; i < _contactsList.length; i++) {
                        if (_contactsList[i].v == deletedContact)
                            _contactsList.splice(i, 1);
                        _setContactsList(_contactsList);
                    }

                    Netdania.Alert.DefaultContactStorage.RemoveContact({v:deletedContact,t:CONTACT_TYPE[0].t});

                    $("#contacts-div").html(RenderContactsList());
                    _setDeleteEvents();
                    customPopup.closePopup();
                }
            });
        });

        $('.nd-pushDevices-check').click(function () {
            for (var i = 0; i < _pushDevices.length; i++) {
                if (_pushDevices[i].id == $(this).attr("device")) {
                    _pushDevices[i].selected = $(this).is(':checked');
                    if (_pushDevices[i].selected) {
                        Netdania.Alert.DefaultContactStorage.AddContact({v: _pushDevices[i].id, t: CONTACT_TYPE[1].t});
                    }
                    else {
                        Netdania.Alert.DefaultContactStorage.RemoveContact({v: _pushDevices[i].id, t: CONTACT_TYPE[1].t});
                    }
                }
            }
        });
    }

    var _renderSelectorControl = function (controlName, fields) {
        var fieldControl = '<select id="' + controlName + '" class="' + controlName + '" >{0}</select>';
        if (fields != undefined && fields.length > 0) {
            for (var i = 0; i < fields.length; i++) {
                fieldControl = fieldControl.replace('{0}', '<option value="' + fields[i].v + '">' + fields[i].t + '</option>{0}')
            }
        }
        return fieldControl.replace('{0}', '');
    }


    this._getAllAlerts = function (callbackFunction) {

            var activeAlerts = new Netdania.JsApi.Alert.AlertObject();
            activeAlerts.USER_ID = config.userID;
            activeAlerts.ALERTS_INCLUDED = true;
            activeAlerts.TIMESCALE_SECONDS = true;
            activeAlerts.ALERT_ID = _currentAlertID;
            activeAlerts.COOKIE = "NDWebsite";



            var req10 = Netdania.JsApi.Request.getReqObjAlertGetUserActiveAlerts(activeAlerts);
            _lastReqID = req10.i;
            _connection.AddRequest(req10);
            _connection.Flush();

            _connection.addListener(Netdania.JsApi.Events.ONALERTGETACTIVE, function (alertAddedResponse) {
                if (alertAddedResponse.type == Netdania.JsApi.Alert.Commands.GET_USER_INFORMATION && _lastReqID == alertAddedResponse.id) {
                    //_connection.removeListener(Netdania.JsApi.Events.ONALERTGETACTIVE);
                    _getAlertsList(alertAddedResponse, callbackFunction);
                }
            });

            _connection.addListener(Netdania.JsApi.Events.ONERROR, function (alertErrorResponse) {
                if (alertErrorResponse && alertErrorResponse.type == Netdania.JsApi.Alert.Commands.ERROR  && _lastReqID == alertErrorResponse.id) {
                    if (callbackFunction) {
                        callbackFunction();
                    }
                    customPopup = new Netdania.Util.PopUp();
                    customPopup.initPopup({ containerName: 'popupContainer', cssClass: "alert-popup" });
                    customPopup.showPopup({ title: 'Loading alerts', message: 'There was an error!', content: '', messageType: 'warning', noCancelButton: true,
                        callbackFunction: function () {
                            customPopup.closePopup();
                        }
                    });
                }
            });
    }


    this._getAllSentAlerts = function (start_date, end_date, callbackFunction) {

            var activeAlerts = new Netdania.JsApi.Alert.AlertObject();
            activeAlerts.USER_ID = config.userID;
            activeAlerts.ALERTS_INCLUDED = true;
            activeAlerts.ALERT_ID = _currentAlertID;
            activeAlerts.TIMESCALE_SECONDS = true;
            activeAlerts.STARTDATE = start_date;
            activeAlerts.ENDDATE = end_date;
            activeAlerts.COOKIE = "NDWebsite";


            //activeAlerts.COOKIE = "CristianCr-AlertGetTriggered"
            var req10 = Netdania.JsApi.Request.getReqObjAlertGetTriggered(activeAlerts);
            _lastReqID = req10.i;
            _connection.AddRequest(req10);
            _connection.Flush();

            _connection.addListener(Netdania.JsApi.Events.ONALERTGETTRIGGERED, function (alertAddedResponse) {
                if (alertAddedResponse.type == Netdania.JsApi.Alert.Commands.GET_TRIGGERED_ALERTS && _lastReqID == alertAddedResponse.id) {
                    _connection.removeListener(Netdania.JsApi.Events.ONALERTGETTRIGGERED);
                    _getSentDeleteAlertsList(alertAddedResponse, callbackFunction, AlertListType.SENT);
                }
            });
    }


    this._getAllDeletedAlerts = function (start_date, end_date, callbackFunction) {

        //USER_ID, USER_GROUP, IS_ALERT_CENTRAL, ALERTS_INCLUDED, USER_DETAILS_INCLUDED
        var activeAlerts = new Netdania.JsApi.Alert.AlertObject();
        activeAlerts.USER_ID = config.userID;
        activeAlerts.ALERTS_INCLUDED = true;
        activeAlerts.TIMESCALE_SECONDS = true;
        activeAlerts.STARTDATE = start_date;
        activeAlerts.ENDDATE = end_date;
        activeAlerts.COOKIE = "NDWebsite";


        //activeAlerts.COOKIE = "CristianCr-AlertGetTriggered"
        var req10 = Netdania.JsApi.Request.getReqObjAlertGetDeleted(activeAlerts);
        var _lastReqID = req10.i;
        //these instruments will be requested from the server after the connection to the server has been succesfully
        _connection.AddRequest(req10);

        _connection.Flush();

        _connection.addListener(Netdania.JsApi.Events.ONALERTGETDELETED, function (alertDeletedResponse) {
            if (alertDeletedResponse.type == Netdania.JsApi.Alert.Commands.GET_DELETED_ALERTS && _lastReqID == alertDeletedResponse.id) {
                _connection.removeListener(Netdania.JsApi.Events.ONALERTGETDELETED);
                _getSentDeleteAlertsList(alertDeletedResponse, callbackFunction, AlertListType.DELETED);
            }
        });
    }

    var _getAlertsList = function (alertResponse, callbackFunction) {
        _alertsList = [];
        try {
            if (alertResponse != undefined && alertResponse.data != undefined && alertResponse.data.f != undefined) {
                for (var i = 0; i < alertResponse.data.f.length; i++) {
                    if (alertResponse.data.f[i].f == Netdania.JsApi.Alert.Fields.DATA_HOLDER) {
                        if (alertResponse.data.f[i].v != undefined && alertResponse.data.f[i].v[0].f != undefined) {
                            for (var j = 0; j < alertResponse.data.f[i].v[0].f.length; j++) {
                                if (alertResponse.data.f[i].v[0].f[j].f == Netdania.JsApi.Alert.Fields.ALERTS_HOLDER) {
                                    if (alertResponse.data.f[i].v[0].f[j].v != undefined) {
                                        for (var k = 0; k < alertResponse.data.f[i].v[0].f[j].v.length; k++) {
                                            if (alertResponse.data.f[i].v[0].f[j].f == Netdania.JsApi.Alert.Fields.ALERTS_HOLDER) {
                                                var config = {};
                                                for (var l = 0; l < alertResponse.data.f[i].v[0].f[j].v[k].f.length; l++) {

                                                    switch (alertResponse.data.f[i].v[0].f[j].v[k].f[l].f) {

                                                        case Netdania.JsApi.Alert.Fields.EMAIL_CONTENT:
                                                        {
                                                            config.emailContent = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }
                                                        case Netdania.JsApi.Alert.Fields.ALERT_ID:
                                                        {
                                                            config.alertID = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }
                                                        case Netdania.JsApi.Alert.Fields.TRIGGER_DATE:
                                                        {
                                                            config.triggerDate = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }
                                                        case Netdania.JsApi.Alert.Fields.ALERT_CONDITION:
                                                        {
                                                            config.alertCondition = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }
                                                        case Netdania.JsApi.Alert.Fields.DAYS_TO_LIVE:
                                                        {
                                                            config.daysToLive = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }
                                                        case Netdania.JsApi.Alert.Fields.EMAIL_ADDRESS:
                                                        {
                                                            config.emailAddress = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }
                                                        case Netdania.JsApi.Alert.Fields.EMAIL_FROM:
                                                        {
                                                            config.emailFrom = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }
                                                        case Netdania.JsApi.Alert.Fields.EMAIL_SUBJECT:
                                                        {
                                                            config.emailSubject = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }
                                                        case Netdania.JsApi.Alert.Fields.YAHOO_ID:
                                                        {
                                                            config.yahooIDs = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }
                                                        case Netdania.JsApi.Alert.Fields.SKYPE_ID:
                                                        {
                                                            config.skypeIDs = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }
                                                        case Netdania.JsApi.Alert.Fields.MSN_ID:
                                                        {
                                                            config.msnIDs = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }
                                                        case Netdania.JsApi.Alert.Fields.PUSH_DEVICES:
                                                        {
                                                            config.pushDevices = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }

                                                        case Netdania.JsApi.Alert.Fields.TIMESCALE_SECONDS:
                                                        {
                                                            config.timescaleSeconds = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }
                                                    }
                                                }

                                                var alertWrapper = new Netdania.Alert.AlertWrapper(config);
                                                alertWrapper.setConditionToObject(config.alertCondition);
                                                _alertsList.push(alertWrapper);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        catch
            (err) {
            Netdania.JsApi.Utilities.LogError("Alert._getAlertsList", err);
        }


        if (callbackFunction) {
            callbackFunction();
        }
    }


    var _getSentDeleteAlertsList = function (alertResponse, callbackFunction, AlertType) {
        _alertsLists[AlertType] = [];
        try {
            if (alertResponse != undefined && alertResponse.data != undefined && alertResponse.data.f != undefined) {
                for (var i = 0; i < alertResponse.data.f.length; i++) {
                    if (alertResponse.data.f[i].f == Netdania.JsApi.Alert.Fields.ALERTS_HOLDER) {
                        for (var j = 0; j < alertResponse.data.f[i].v.length; j++) {
                            if (alertResponse.data.f[i].v[j].f != undefined) {

                                var config = {};

                                for (var k = 0; k < alertResponse.data.f[i].v[j].f.length; k++) {

                                    switch (alertResponse.data.f[i].v[j].f[k].f) {

                                        case Netdania.JsApi.Alert.Fields.EMAIL_CONTENT:
                                        {
                                            config.emailContent = alertResponse.data.f[i].v[j].f[k].v[0];
                                            break;
                                        }
                                        case Netdania.JsApi.Alert.Fields.ALERT_ID:
                                        {
                                            config.alertID = alertResponse.data.f[i].v[j].f[k].v[0];
                                            break;
                                        }
                                        case Netdania.JsApi.Alert.Fields.TRIGGER_DATE:
                                        {
                                            config.triggerDate = alertResponse.data.f[i].v[j].f[k].v[0];
                                            break;
                                        }
                                        case Netdania.JsApi.Alert.Fields.ALERT_CONDITION:
                                        {
                                            config.alertCondition = alertResponse.data.f[i].v[j].f[k].v[0];
                                            break;
                                        }
                                        case Netdania.JsApi.Alert.Fields.DAYS_TO_LIVE:
                                        {
                                            config.daysToLive = alertResponse.data.f[i].v[j].f[k].v[0];
                                            break;
                                        }
                                        case Netdania.JsApi.Alert.Fields.EMAIL_ADDRESS:
                                        {
                                            config.emailAddress = alertResponse.data.f[i].v[j].f[k].v[0];
                                            break;
                                        }
                                        case Netdania.JsApi.Alert.Fields.EMAIL_FROM:
                                        {
                                            config.emailFrom = alertResponse.data.f[i].v[j].f[k].v[0];
                                            break;
                                        }
                                        case Netdania.JsApi.Alert.Fields.EMAIL_SUBJECT:
                                        {
                                            config.emailSubject = alertResponse.data.f[i].v[j].f[k].v[0];
                                            break;
                                        }
                                        case Netdania.JsApi.Alert.Fields.YAHOO_ID:
                                        {
                                            config.yahooIDs = alertResponse.data.f[i].v[j].f[k].v[0];
                                            break;
                                        }
                                        case Netdania.JsApi.Alert.Fields.SKYPE_ID:
                                        {
                                            config.skypeIDs = alertResponse.data.f[i].v[j].f[k].v[0];
                                            break;
                                        }
                                        case Netdania.JsApi.Alert.Fields.MSN_ID:
                                        {
                                            config.msnIDs = alertResponse.data.f[i].v[j].f[k].v[0];
                                            break;
                                        }
                                        case Netdania.JsApi.Alert.Fields.PUSH_DEVICES:
                                        {
                                            config.pushDevices = alertResponse.data.f[i].v[j].f[k].v[0];
                                            break;
                                        }

                                        case Netdania.JsApi.Alert.Fields.DELETION_DATE:
                                        {
                                            config.deletionDate = alertResponse.data.f[i].v[j].f[k].v[0];
                                            break;
                                        }

                                        case Netdania.JsApi.Alert.Fields.DELETION_REASON:
                                        {
                                            config.deletionReason = alertResponse.data.f[i].v[j].f[k].v[0];
                                            break;
                                        }
                                    }
                                }

                                var alertWrapper = new Netdania.Alert.AlertWrapper(config);
                                alertWrapper.setConditionToObject(config.alertCondition);
                                _alertsLists[AlertType].push(alertWrapper);
                            }
                        }
                    }
                }
            }
        }
        catch
            (err) {
            Netdania.JsApi.Utilities.LogError("Alert._getSentDeleteAlertsList", err);
        }

        if (callbackFunction) {
            callbackFunction();
        }
        _self._hideLoader();
    }


    this.addAlertObjectFromResponse = function (alertResponse) {
        var config = {};
        config.emailContent = alertResponse.get(Netdania.JsApi.Alert.Fields.EMAIL_CONTENT);
        config.alertID = alertResponse.get(Netdania.JsApi.Alert.Fields.ALERT_ID);
        config.triggerDate = alertResponse.get(Netdania.JsApi.Alert.Fields.TRIGGER_DATE);
        config.alertCondition = alertResponse.get(Netdania.JsApi.Alert.Fields.ALERT_CONDITION);
        config.daysToLive = alertResponse.get(Netdania.JsApi.Alert.Fields.DAYS_TO_LIVE);
        config.emailAddress = alertResponse.get(Netdania.JsApi.Alert.Fields.EMAIL_ADDRESS);
        config.emailFrom = alertResponse.get(Netdania.JsApi.Alert.Fields.EMAIL_FROM);
        config.emailSubject = alertResponse.get(Netdania.JsApi.Alert.Fields.EMAIL_SUBJECT);
        config.yahooIDs = alertResponse.get();
        config.skypeIDs = alertResponse.get(Netdania.JsApi.Alert.Fields.YAHOO_ID);
        config.msnIDs = alertResponse.get(Netdania.JsApi.Alert.Fields.MSN_ID);

        var alertWrapper = new Netdania.Alert.AlertWrapper(config);
        alertWrapper.setConditionToObject(config.alertCondition);
        _alertsList.push(alertWrapper);
    }

    this._refreshGrid = function () {
        if (_alertsGridContainer)
        {
        _self._showLoader();
        _self._getAllAlerts(_self.renderGrid);
        }
    }

    this.renderGrid = function () {

        try {
            if (_alertsGridContainer) {
                $("#" + _alertsGridContainer).flexAddData(formatCustomerResultsActive(eval(_alertsList)));
                _addDeleteEventsOnGridButtons(eval(_alertsList));
                _addEditEventsOnGridButtons(eval(_alertsList));
            }
        }
        catch (ex) {
            Netdania.JsApi.Utilities.LogError("Alerts.renderGrid", ex);
        }
        _self._hideLoader();

    }

    this.renderSentAlertsGrid = function () {

        try {
            $("#" + _alertsSentGridContainer).flexAddData(formatCustomerResultsSent(eval(_alertsLists[AlertListType.SENT])));
        }
        catch (ex) {
            Netdania.JsApi.Utilities.LogError("Alerts.renderSentAlertsGrid", ex);
        }
        _self._hideLoader();
    }

    this.renderDeletedAlertsGrid = function () {

        try {
            $("#" + _alertsDeletedGridContainer).flexAddData(formatCustomerResultsDeleted(eval(_alertsLists[AlertListType.DELETED])));
        }
        catch (ex) {
            Netdania.JsApi.Utilities.LogError("Alerts.renderDeletedAlertsGrid", ex);
        }
        _self._hideLoader();
    }

    this._refreshSentAlertsGrid = function () {
        try {
            if (_alertsSentGridContainer)
            _self._getAllSentAlerts(_getSentStartDate(), _getSentEndDate(), _self.renderSentAlertsGrid);
        }
        catch (ex) {
            Netdania.JsApi.Utilities.LogError("Alerts._refreshSentAlertsGrid", ex);
        }
    }

    var _getSentStartDate = function () {

        var date = $('#' + _alertsSentStartDateInput).datepicker("getDate");
        if (date != null) {
            return date.getTime();
        }
        else {
            return (new Date(new Date() - 30 * _oneday).getTime());
        }
    };

    var _getSentEndDate = function () {
        var date = $('#' + _alertsSentEndDateInput).datepicker("getDate");
        if (date != null) {
            return new Date(new Date(date).getTime() + _oneday).getTime();
        }
        else {
            return (new Date(new Date() + _oneday).getTime());
        }
    };

    var _getDeletedStartDate = function () {

        var sDate = $('#' + _alertsDeletedStartDateInput).datepicker("getDate");
        if (sDate != null) {
            return sDate.getTime();
        }
        else {
            sDate = new Date(new Date() - 30 * _oneday).getTime();
            return sDate;
        }
    };

    var _getDeletedEndDate = function () {
        var date = $('#' + _alertsDeletedEndDateInput).datepicker("getDate");
        if (date != null) {
            return new Date(new Date(date).getTime() + _oneday).getTime();
        }
        else {
            return (new Date(new Date() + _oneday).getTime());
        }
    };

    this._refreshDeletedAlertsGrid = function () {
        if (_alertsDeletedGridContainer)
        _self._getAllDeletedAlerts(_getDeletedStartDate(), _getDeletedEndDate(), _self.renderDeletedAlertsGrid);
    }


    var setAlertsTable = function (objectList) {
        var html = "<table border='0' width='100%'>";
        var th = "<tr>" +
            "<th>Instrument</th>" +
            "<th>Condition</th>" +
            "<th>Message</th>" +
            "<th>Creation date</th>" +
            "<th>Expire date</th>" +
            "<th></th>" +
            "<th></th>" +
            "</tr>";
        html += th;
        for (i = 0; i < objectList.length; i++) {

            try {
                var item = objectList[i].getAlertObject();
                html += "<tr id='td_" + item.ALERT_ID + "'>";
                var alert = item.Condition ? item : objectList[i];
                var operator = alert.Condition.getRelationalOperator().v ? alert.Condition.getRelationalOperator().v : alert.Condition.getRelationalOperator();
                // name
                html += "<td>" + alert.Instrument.getName() + "</td>";
                // field condition value 
                html += "<td>" + Netdania.Alert.getFieldNameByValue(alert.Instrument.getField()) + ' ' + operator + ' ' + alert.Condition.getAlertValue() + "</td>";
                // message

                html += "<td>" + ((item.EMAIL_CONTENT.split("Message: ")[1] === undefined || item.EMAIL_CONTENT.split("Message: ")[1] == '') ? '' : item.EMAIL_CONTENT.split("Message: ")[1]) + "</td>";
                // creation - expire date
                var creationDate = new Date(alert.Condition.getTimestamp() * 1);
                var expireDate = new Date();
                expireDate.setTime(creationDate.getTime() + (item.DAYS_TO_LIVE * _oneday));
                html += "<td>" + creationDate.format(config.dateFormat) + "</td>";
                html += "<td>" + expireDate.format(config.dateFormat) + "</td>";
                // edit button
                html += '<td><input type="button" id="btnEdit' + item.ALERT_ID + '" value="Edit" /></td>';
                // delete button
                html += '<td><input type="button" id="btnDelete' + item.ALERT_ID + '" value="Delete" /></td>';

            }

            catch (exIter) {
                Netdania.JsApi.Utilities.LogError("Alerts.setAlertsTable", exIter);
            }
            html += "</tr>";
        }
        html += "</table>";
        return html;
    }

    var _addDeleteEventsOnGridButtons = function (objectList) {
        for (i = 0; i < objectList.length; i++) {
            var item = objectList[i].getAlertObject();

            var deleteButton = $("#btnDelete" + item.ALERT_ID).button();
            deleteButton.click([item.ALERT_ID], function (eventObject) {
                deleteAlert(eventObject.data[0]);
            });
        }
    }

    var _addEditEventsOnGridButtons = function (objectList) {
        for (i = 0; i < objectList.length; i++) {
            var item = objectList[i].getAlertObject();
            var alertType = Netdania.Alert.Types.PRICE;

            if (item.Instrument) {
                alertType = item.Instrument.getType();
            }
            else {
                objectList[i].setConditionToObject(item.ALERT_CONDITION);
                item = objectList[i].getAlertObject();
                if (item.Instrument) {
                    alertType = item.Instrument.getType();
                }
            }
            var editButton;

            if (alertType === Netdania.Alert.Types.PRICE) {

                editButton = $("#btnEdit" + item.ALERT_ID).button();
                editButton.click([item.ALERT_ID], function (eventObject) {
                    editAlert(eventObject.data[0]);
                });
            } else {
                editButton = $("#btnEdit" + item.ALERT_ID).button({ disabled: true });
            }
        }
    }

    var formatCustomerResultsActive = function (objectList) {

        var rows = Array();
        try {
            for (i = 0; i < objectList.length; i++) {
                try {
                    var item = objectList[i].getAlertObject();
                    var alert = item.Condition ? item : objectList[i];
                    var operator = alert.Condition.getRelationalOperator().v ? alert.Condition.getRelationalOperator().v : alert.Condition.getRelationalOperator();
                    //Do something here with the link
                    var deleteBtnID = "btnDelete" + item.ALERT_ID;
                    var editBtnID = "btnEdit" + item.ALERT_ID;
                    var creationDate = new Date(alert.Condition.getTimestamp() * 1);
                    var expireDate = new Date();
                    expireDate.setTime(creationDate.getTime() + (item.DAYS_TO_LIVE * _oneday));
                    if (_filter) {
                        if (_filter[0].value == alert.Instrument.getSymbol()) {

                            rows.push({ cell: [
                                alert.Instrument.getName(),
                                Netdania.Alert.getFieldNameByValue(alert.Instrument.getField()) + ' ' + operator + ' ' + alert.Condition.getAlertValue() ,
                                alert.Instrument.getType() +   formatTimescale(alert.Condition.getTimeScale()),
                                (item.EMAIL_CONTENT !== undefined ? item.EMAIL_CONTENT.split("Message: ")[1] === undefined ? '' : item.EMAIL_CONTENT.split("Message: ")[1] : ""),                               
                                creationDate.format(config.dateFormat),
                                creationDate.getTime(),
                                expireDate.format(config.dateFormat),
                                expireDate.getTime(),
                                '<button title="Edit alert" class="Edit" type="button" id="' + editBtnID + '" value="Edit" />' +
                                '<button title="Delete alert" class="Delete" type="button" id="' + deleteBtnID + '" value="Delete" />']
                            });
                        }
                    }
                    else {
                        rows.push({ cell: [
                            alert.Instrument.getName(),
                            Netdania.Alert.getFieldNameByValue(alert.Instrument.getField()) + ' ' + operator + ' ' + alert.Condition.getAlertValue(),
                            alert.Instrument.getType() + formatTimescale(alert.Condition.getTimeScale()),
                            (item.EMAIL_CONTENT !== undefined ? item.EMAIL_CONTENT.split("Message: ")[1] === undefined ? '' : item.EMAIL_CONTENT.split("Message: ")[1] : ""),                            
                            creationDate.format(config.dateFormat),
                            creationDate.getTime(),
                            expireDate.format(config.dateFormat),
                            expireDate.getTime(),
                            '<button title="Edit alert" class="Edit" type="button" id="' + editBtnID + '" value="Edit" /> ' +
                            '<button title="Delete alert" class="Delete" type="button" id="' + deleteBtnID + '" value="Delete" />']
                        });
                    }
                }
                catch (exIter) {
                    Netdania.JsApi.Utilities.LogError("Alerts.formatCustomerResultsActive", exIter);
                }
            }

            return {
                total: rows.length,
                page: 1,
                rows: rows
            };
        }
        catch (ex) {
            Netdania.JsApi.Utilities.LogError("Alerts.formatCustomerResultsActive.P", ex);
        }
    }

    var formatCustomerResultsSent = function (objectList) {

        var rows = Array();
        try {
            for (i = 0; i < objectList.length; i++) {
                try {
                    var item = objectList[i].getAlertObject();
                    var alert = item.Condition ? item : objectList[i];
                    var operator = alert.Condition.getRelationalOperator().v ? alert.Condition.getRelationalOperator().v : alert.Condition.getRelationalOperator();
                    var triggeredDate = new Date(item.TRIGGER_DATE)

                    rows.push({ cell: [
                        triggeredDate.format(config.dateFormat),
                        triggeredDate.getTime(),
                        alert.Instrument.getName(),
                        Netdania.Alert.getFieldNameByValue(alert.Instrument.getField()) + ' ' + operator + ' ' + alert.Condition.getAlertValue(),
                        alert.Instrument.getType() +  formatTimescale(alert.Condition.getTimeScale()),
                        item.EMAIL_CONTENT.split("Message: ")[1] === undefined ? '' : item.EMAIL_CONTENT.split("Message: ")[1]]
                    });

                }
                catch (exIter) {
                    Netdania.JsApi.Utilities.LogError("Alerts.formatCustomerResultsSent", exIter);
                }
            }

            return {
                total: rows.length,
                page: 1,
                rows: rows
            };
        }
        catch (ex) {
            Netdania.JsApi.Utilities.LogError("Alerts.formatCustomerResultsSent.P", ex);
        }
    }

    var formatCustomerResultsDeleted = function (objectList) {

        var rows = Array();
        try {
            for (i = 0; i < objectList.length; i++) {
                try {
                    var item = objectList[i].getAlertObject();
                    var alert = item.Condition ? item : objectList[i];
                    var operator = alert.Condition.getRelationalOperator().v ? alert.Condition.getRelationalOperator().v : alert.Condition.getRelationalOperator();
                    var deletionDate = new Date(item.DELETION_DATE)
                    var creationDate = new Date(alert.Condition.getTimestamp() * 1);
                    var expireDate = new Date();
                    var deleteReason = AlertDeleteReason.Name(item.DELETION_REASON);

                    expireDate.setTime(creationDate.getTime() + (item.DAYS_TO_LIVE * _oneday));

                    rows.push({ cell: [
                        alert.Instrument.getName(),
                        Netdania.Alert.getFieldNameByValue(alert.Instrument.getField()) + ' ' + operator + ' ' + alert.Condition.getAlertValue(),
                        alert.Instrument.getType() +  formatTimescale(alert.Condition.getTimeScale()),
                        item.EMAIL_CONTENT.split("Message: ")[1] === undefined ? '' : item.EMAIL_CONTENT.split("Message: ")[1],
                        creationDate.format(config.dateFormat),
                        creationDate.getTime(),
                        expireDate.format(config.dateFormat),
                        expireDate.getTime(),
                        deletionDate.format(config.dateFormat),
                        deletionDate.getTime(),
                        deleteReason]
                    });
                }
                catch (exIter) {
                    Netdania.JsApi.Utilities.LogError("Alerts.formatCustomerResultsDeleted", exIter);
                }
            }

            return {
                total: rows.length,
                page: 1,
                rows: rows
            };
        }
        catch (ex) {
            Netdania.JsApi.Utilities.LogError("Alerts.formatCustomerResultsDeleted.P", ex);
        }
    }

    var formatTimescale = function (timescale) {
        var M_5SEC = 5;
        var M_10SEC = 10;
        var M_15SEC = 15;
        var M_30SEC = 30;
        var M_1MIN = 60;
        var M_2MIN = 120;
        var M_3MIN = 180;
        var M_4MIN = 240;
        var M_5MIN = 300;
        var M_10MIN = 600;
        var M_15MIN = 900;
        var M_30MIN = 1800;
        var M_1HOUR = 3600;
        var M_2HOURS = 7200;
        var M_3HOURS = 10800;
        var M_4HOURS = 14400;
        var M_6HOURS = 21600;
        var M_8HOURS = 28800;
        var M_12HOURS = 43200;
        var M_DAILY = 86400;
        var M_WEEKLY = 604800;
        var M_MONTHLY = 2592000;

        var measureUnit = "";

        try {
            switch (parseInt(timescale)) {
                case M_5SEC:
                case M_10SEC:
                case M_15SEC:
                case M_30SEC:
                    measureUnit = "</br>" + timescale + " seconds";
                    break;
                case M_1MIN:
                    measureUnit = "</br>" + "1 minute";
                    break;
                case M_2MIN:
                case M_3MIN:
                case M_4MIN:
                case M_5MIN:
                case M_10MIN:
                case M_15MIN:
                case M_30MIN:
                    measureUnit = "</br>" + timescale / 60 + " minutes";
                    break;
                case M_1HOUR:
                    measureUnit = "</br>" + "1 hour";
                    break;
                case M_2HOURS:
                case M_3HOURS:
                case M_4HOURS:
                case M_6HOURS:
                case M_8HOURS:
                case M_12HOURS:
                    measureUnit = "</br>" + timescale / 3600 + " hours";
                    break;
                case M_DAILY:
                    measureUnit = "</br>" + "Daily";
                    break;
                case M_WEEKLY:
                    measureUnit = "</br>" + "Weekly";
                    break;
                case M_MONTHLY:
                    measureUnit = "</br>" + "Monthly";
                    break;
            }
        }
        catch (ex) {
            Netdania.JsApi.Utilities.LogError("Alert.formatTimescale",ex);
        }
        return  measureUnit;
    }


    var fillAlertForm = function (alertObject, tmpAlertObj) {

        $("#nd-alert_dialog-form").dialog('option', 'title', 'Edit alert');
        $("#nd-alert_dialog-form").dialog("open");
        var alert = alertObject.Condition ? alertObject : tmpAlertObj;
        var operator = alert.Condition.getRelationalOperator().v ? alert.Condition.getRelationalOperator().v : alert.Condition.getRelationalOperator();
        _currentAlertID = alertObject.ALERT_ID;
        $("#" + config.operatorsControlName).val(operator);
        _setOperator({ t: operator, v: operator });
        $("#" + config.valueAlertName).val(alert.Condition.getAlertValue());
        $("#" + config.valueAlertName).change();
        $("#" + config.fieldsControlName).val(alert.Instrument.getField());
        $("#" + config.messageAlertName).val(alertObject.EMAIL_CONTENT !== undefined ? alertObject.EMAIL_CONTENT.split("Message: ")[1] === undefined ? '' : alertObject.EMAIL_CONTENT.split("Message: ")[1] : "");
        var creationTimestamp = (alert.Condition.getTimestamp() * 1);
        var createdDate = new Date(creationTimestamp);

        $("#" + _creationDate).val(creationTimestamp);

        var expireDate = new Date();
        expireDate.setTime(createdDate.getTime() + alert.Condition.getTimeout() * _oneday);

        $("#" + _expDate).val(expireDate.format(config.dateFormat));
        _currentQuoteList = Netdania.Alert.QuoteList({ operatorID: config.operatorsControlName, initialUpdate: false, updateField: 'nd-txtValueAlert', symbol: alert.Instrument.getSymbol(), provider: alert.Instrument.getProvider(), containerId: 'nd-instrumentBidAsk', selectedField: alert.Instrument.getField(), callbackFunction: _self.updateAllowedFields, ddHtmlElementID: "nd-alert-dd" });

        $("#" + config.operatorsControlName).change(function () {
           validateAlertValue(false);
        });
        $("#" + config.valueAlertName).keyup(function () {
            validateAlertValue(false);
        }).focus(function () {
                validateAlertValue(false);
            }).focusout(function () {
                validateAlertValue(false);
            });
        $("#" + config.emailInputName).keyup(validateEmail).focus(validateEmail);


        //add email contacts
        if (alertObject.EMAIL_ADDRESS) {
            var contacts = alertObject.EMAIL_ADDRESS.split(';');
            pushContactToList(contacts, CONTACT_TYPE[0].t);
        }

        for (var j = 0; j < _pushDevices.length; j++) {
            _pushDevices[j].selected = false;
        }
        if (alertObject.PUSH_DEVICES) {
            var selectedDevices = alertObject.PUSH_DEVICES.split(',');
            _self.addSelectedDevices(selectedDevices);
        }

        $("#contacts-div").html("");

        $("#contacts-div").html(RenderContactsList());
        _setDeleteEvents();
    }

    var clearAlertForm = function () {

        $("#" + config.operatorsControlName).val('');
        _setOperator(RESTRICTED_OPERATORS[0]);
        $("#" + config.valueAlertName).val('');
        $("#" + config.messageAlertName).val('');
    }

    var editAlert = function (alertID) {
        var alertObject;
        var tmpAlertObj;
        for (var i = 0; i < _alertsList.length; i++) {
            if (alertID == _alertsList[i].getAlertObject().ALERT_ID) {
                tmpAlertObj = _alertsList[i];
                alertObject = _alertsList[i].getAlertObject();
                break;
            }
            ;
        }
        _contactsList = [];
        _setContactsList( _contactsList);
        if (alertObject !== undefined && alertObject.Instrument !== undefined) {
            _selectedAlertObject = alertObject;
        }
        else {
            _selectedAlertObject = tmpAlertObj;
        }
        if (alertObject !== undefined)
        {
            fillAlertForm(alertObject, tmpAlertObj);
        }
        else

        {_self._refreshGrid();}

    }

    var pushContactToList = function (contacts, contactsType) {
        var _user = new Netdania.Util.User();
        var userID = _user.getUserId();

        var contactAlreadyAdded = false;
        _contactsList = _getContactsList();
        for (var i = 0; i < contacts.length; i++) {
            contactAlreadyAdded = false;
            for (var j = 0; j < _contactsList.length; j++) {
                if (_contactsList[j].v == contacts[i] && _contactsList[j].t == contactsType) {
                    contactAlreadyAdded = true;
                    break;
                }
            }
            if (!contactAlreadyAdded) {
                var del = true;
                if (contacts[i] == userID)
                    del = false;
                _contactsList.push({ v: contacts[i], t: contactsType })
                _setContactsList(_contactsList);
            }
        }
    }

    this.addSelectedDevices = function (selectedDevices) {
        for (var i = 0; i < selectedDevices.length; i++) {
            for (var j = 0; j < _pushDevices.length; j++) {
                if (_pushDevices[j].id == selectedDevices[i]) {
                    _pushDevices[j].selected = true;
                    break;
                }
            }
        }
    };

    var deleteAlert = function (alertID) {
        customPopup = new Netdania.Util.PopUp();
        customPopup.initPopup({ containerName: 'popupContainer', cssClass: "alert-popup" });
        customPopup.showPopup({ title: 'Delete confirmation', message: 'Are you sure you want to delete this alert?', showButtons: true, messageType: 'successfull',
            callbackFunction: function () {

                var alertObject;
                for (var i = 0; i < _alertsList.length; i++) {
                    if (alertID == _alertsList[i].getAlertObject().ALERT_ID) {
                        alertObject = _alertsList[i].getAlertObject();
                        _alertsList.splice(i, 1);
                        break;
                    }
                }
                _self._showLoader();

                alertObject.DELETION_REASON = AlertDeleteReason.DELETED_BY_USER;

                var req10 = Netdania.JsApi.Request.getReqObjAlertDelete(alertObject);
                _deleteReqID = req10.i;
                _connection.AddRequest(req10);
                _connection.Flush();

                _connection.addListener(Netdania.JsApi.Events.ONALERTDELETE, function (alertAddedResponse) {



                    Netdania.Alert.AlertStorage.RemoveAlert(alertID);
                    //Netdania.Alert.RemoveAlertFromMonitoredAlertsCookie(alertID);
                    _self.renderGrid();
                    customPopup.closePopup();
                    _self._hideLoader();
                    _deleteReqID = 0;
                });

                _connection.addListener(Netdania.JsApi.Events.ONERROR, function (alertErrorResponse) {
                    if (alertErrorResponse && alertErrorResponse.type == Netdania.JsApi.Alert.Commands.ERROR  && _deleteReqID == alertErrorResponse.id) {

                        _self.renderGrid();
                        customPopup.closePopup();
                        _self._hideLoader();
                        _deleteReqID = 0;

                        customPopup = new Netdania.Util.PopUp();
                        customPopup.initPopup({ containerName: 'popupContainer', cssClass: "alert-popup" });
                        customPopup.showPopup({ title: 'Delete alert', message: 'There was an error!', content: '', messageType: 'warning', noCancelButton: true,
                            callbackFunction: function () {
                                customPopup.closePopup();
                            }
                        });
                    }
                });
            }
        });
    }


    // Public methods START

    this._showLoader = function () {
        $("#circularGLoaderDiv")[0].style.display = "block";
    }
    this._hideLoader = function () {
        $("#circularGLoaderDiv")[0].style.display = "none";
    }

    // Public methods END

    function defaultContactsList() {
        var contactsList = [];

        /*
        Skip adding default email
        var _user = new Netdania.Util.User();
        var userID = _user.getUserId();
        contactsList.push({ v: userID, t: 'email'});
          */

        var storageContacts  = Netdania.Alert.DefaultContactStorage.GetContacts({t:CONTACT_TYPE[0].t});
        if (storageContacts.length >0 ){
            contactsList = contactsList.concat(storageContacts);
        }


        return contactsList;
    }

    /**
     * Set selected push devices from the cookie
     */
    function defaultPushDevices() {
        var storageDefaultContacts = Netdania.Alert.DefaultContactStorage.GetContacts({t: CONTACT_TYPE[1].t});

        $.each(storageDefaultContacts, function (c) {
            $.each(_pushDevices, function (i) {
                if (_pushDevices[i].id == storageDefaultContacts[c].v) {
                    _pushDevices[i].selected = true;
                    return false;
                }
            });
        });
    }



    function checkIfContactAlreadyAdded(contact, contactsList) {

        for (var i = 0; i < contactsList.length; i++) {
            if (contactsList[i].v == contact[0] && contactsList[i].t == contact[1])
                return true;
        }
        return false;
    }
}

Netdania.Alert.GetAllAlerts = function (userID) {
    var connectionConfig = {
        pollingInterval: Netdania.JsApi.PoolingInterval.AUTO, //milliseconds
        host: Netdania.Alert.Config.HostAlerts,
        behavior: Netdania.JsApi.ConnectionType.AUTO, //LONG POLLING
        type: 1 //the server response format: JSONP
    }
    //create a json connection object
    var _connection = new Netdania.JsApi.JSONConnection(connectionConfig);

    Netdania.Alert.GetPushDevices(userID);

    //get all user alerts
    var activeAlerts = new Netdania.JsApi.Alert.AlertObject();
    activeAlerts.USER_ID = userID;
    activeAlerts.ALERTS_INCLUDED = true;
    activeAlerts.TIMESCALE_SECONDS = true;
    activeAlerts.ALERT_ID = 0;
    activeAlerts.COOKIE = "NDWebsite";


    var req01 = Netdania.JsApi.Request.getReqObjAlertAddUser(activeAlerts);
    _connection.AddRequest(req01);
    _connection.Flush();


    var req10 = Netdania.JsApi.Request.getReqObjAlertGetUserActiveAlerts(activeAlerts);
    _lastReqID = req10.i;
    _connection.AddRequest(req10);
    _connection.Flush();

    _connection.addListener(Netdania.JsApi.Events.ONALERTGETACTIVE, function (alertAddedResponse) {
        if (alertAddedResponse.type == Netdania.JsApi.Alert.Commands.GET_USER_INFORMATION && _lastReqID == alertAddedResponse.id) {
            //_connection.removeListener(Netdania.JsApi.Events.ONALERTGETACTIVE);
            //save alerts on cookie
            //var cookies = Netdania.Util.Cookies();
            //cookies.setCookie("monitoredAlerts", _getAlertsList(alertAddedResponse));
            var alerts = _getAlertsList(alertAddedResponse);
            Netdania.Alert.AlertStorage.AddAllAlerts(alerts);
            Netdania.Alert.MonitorUserAlerts(userID);
        }
    });


    _connection.addListener(Netdania.JsApi.Events.ONERROR, function (alertErrorResponse) {
        if (alertErrorResponse && alertErrorResponse.type == Netdania.JsApi.Alert.Commands.ERROR && _lastReqID == alertErrorResponse.id) {
            _connection.removeListener(Netdania.JsApi.Events.ONALERTGETACTIVE);
        }
    });

    var _getAlertsList = function (alertResponse) {
        _alertsObjectList = [];
        try {
            if (alertResponse != undefined && alertResponse.data != undefined && alertResponse.data.f != undefined) {
                for (var i = 0; i < alertResponse.data.f.length; i++) {
                    if (alertResponse.data.f[i].f == Netdania.JsApi.Alert.Fields.DATA_HOLDER) {
                        if (alertResponse.data.f[i].v != undefined && alertResponse.data.f[i].v[0].f != undefined) {
                            for (var j = 0; j < alertResponse.data.f[i].v[0].f.length; j++) {
                                if (alertResponse.data.f[i].v[0].f[j].f == Netdania.JsApi.Alert.Fields.ALERTS_HOLDER) {
                                    if (alertResponse.data.f[i].v[0].f[j].v != undefined) {
                                        for (var k = 0; k < alertResponse.data.f[i].v[0].f[j].v.length; k++) {
                                            if (alertResponse.data.f[i].v[0].f[j].f == Netdania.JsApi.Alert.Fields.ALERTS_HOLDER) {
                                                var config = {};
                                                for (var l = 0; l < alertResponse.data.f[i].v[0].f[j].v[k].f.length; l++) {

                                                    switch (alertResponse.data.f[i].v[0].f[j].v[k].f[l].f) {
                                                        case Netdania.JsApi.Alert.Fields.ALERT_ID:
                                                        {
                                                            config.alertID = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }

                                                        case Netdania.JsApi.Alert.Fields.ALERT_CONDITION:
                                                        {
                                                            config.alertCondition = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }
                                                        case Netdania.JsApi.Alert.Fields.EMAIL_CONTENT:
                                                        {
                                                            config.emailContent = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }

                                                        case Netdania.JsApi.Alert.Fields.DAYS_TO_LIVE:
                                                        {
                                                            config.daysToLive = alertResponse.data.f[i].v[0].f[j].v[k].f[l].v[0];
                                                            break;
                                                        }
                                                    }
                                                }

                                                var alertWrapper = new Netdania.Alert.AlertWrapper(config);
                                                alertWrapper.setConditionToObject(config.alertCondition);
                                                _alertsObjectList.push(alertWrapper.GetStorageObject());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        catch (err) {
            Netdania.JsApi.Utilities.LogError("Netdania.Alert.GetAllAlerts Error", err);
        }
        return _alertsObjectList;
    }
}

Netdania.Alert.MonitorUserAlerts = function (userID) {

    Netdania.Alert.AlertMonitor.SetActiveAlerts(Netdania.Alert.AlertStorage.GetAllAlerts());

    var _getAlertFromServer = function (alertID, notificationType) {
        try {

            //get all user alerts
            var activeAlerts = new Netdania.JsApi.Alert.AlertObject();
            activeAlerts.USER_ID = userID;
            activeAlerts.ALERT_ID = alertID;
            activeAlerts.TIMESCALE_SECONDS = true;
            activeAlerts.COOKIE = "NDWebsite";

            var req10 = Netdania.JsApi.Request.getReqObjAlertGet(activeAlerts);
            var _lastReqID = req10.i;
            _connection.AddRequest(req10);
            _connection.Flush();

            _connection.addListener(Netdania.JsApi.Events.ONALERTGET, function (alertResponse) {

                try {
                    if (_lastReqID == alertResponse.id) {

                        var config = {};
                        for (var i = 0; i < alertResponse.data.f.length; i++) {

                            switch (alertResponse.data.f[i].f) {
                                case Netdania.JsApi.Alert.Fields.ALERT_ID:
                                {
                                    config.alertID = alertResponse.data.f[i].v[0];
                                    break;
                                }

                                case Netdania.JsApi.Alert.Fields.ALERT_CONDITION:
                                {
                                    config.alertCondition = alertResponse.data.f[i].v[0];
                                    break;
                                }
                                case Netdania.JsApi.Alert.Fields.EMAIL_CONTENT:
                                {
                                    config.emailContent = alertResponse.data.f[i].v[0];
                                    break;
                                }

                                case Netdania.JsApi.Alert.Fields.DAYS_TO_LIVE:
                                {
                                    config.daysToLive = alertResponse.data.f[i].v[0];
                                    break;
                                }

                                case Netdania.JsApi.Alert.Fields.EMAIL_ADDRESS:
                                {
                                    config.emailAddress = alertResponse.data.f[i].v[0];
                                    break;
                                }

                                case Netdania.JsApi.Alert.Fields.TIMESCALE_SECONDS:
                                {
                                    config.timescaleSeconds = alertResponse.data.f[i].v[0];
                                    break;
                                }

                                case Netdania.JsApi.Alert.Fields.PUSH_DEVICES:
                                {
                                    config.pushDevices = alertResponse.data.f[i].v[0];
                                    break;
                                }
                            }
                        }

                        var alertWrapper = new Netdania.Alert.AlertWrapper(config);
                        alertWrapper.setConditionToObject(config.alertCondition);
                        var alertObject = alertWrapper.getAlertObject();


                        switch (notificationType) {
                            case Netdania.JsApi.Alert.NotificationTypes.ALERT_EDITED:
                            {
                                Netdania.Alert.AlertStorage.ReplaceAlert(alertWrapper.GetStorageObject());
                                if (globalAlertObject) {
                                    globalAlertObject._replaceAlert(alertWrapper);
                                    globalAlertObject.renderGrid();
                                }
                                break;
                            }
                            case Netdania.JsApi.Alert.NotificationTypes.ALERT_ADDED:
                            {
                                Netdania.Alert.AlertStorage.SaveAlert(alertWrapper.GetStorageObject());
                                Netdania.Alert.AlertMonitor.AlertAdded(alertWrapper.GetStorageObject());

                                if (globalAlertObject) {
                                    globalAlertObject._addAlertActive(alertWrapper);
                                    globalAlertObject.renderGrid();
                                }

                                break;
                            }
                        }
                    }
                } catch
                    (err) {
                    Netdania.JsApi.Utilities.LogError("MonitorUserAlerts.ONALERTGET Event error", err);
                }
            });
        }
        catch
            (err) {
            Netdania.JsApi.Utilities.LogError("MonitorUserAlerts._getAlertFromServer function error", err);
        }
    }

    try {

        var activeAlerts = new Netdania.JsApi.Alert.AlertObject();
        activeAlerts.USER_ID = userID;
        activeAlerts.TIMESCALE_SECONDS = true;
        activeAlerts.COOKIE = "NDWebsite";

        var connectionConfig = {
            pollingInterval: Netdania.JsApi.PoolingInterval.AUTO, //milliseconds
            host: Netdania.Alert.Config.HostAlerts,
            behavior: Netdania.JsApi.ConnectionType.AUTO, //LONG POLLING
            type: 1 //the server response format: JSONP
        }
        //create a json connection object
        var _connection = new Netdania.JsApi.JSONConnection(connectionConfig);

        var req10 = Netdania.JsApi.Request.getReqObjAlertAddUser(activeAlerts);
        _connection.AddRequest(req10);
        _connection.Flush();

        var req11 = Netdania.JsApi.Request.getReqObjAlertMonitorUser(activeAlerts);
        _connection.AddRequest(req11);
        _connection.Flush();

        var req12 = Netdania.JsApi.Request.getReqObjAlertMonitorUserActivity(activeAlerts);
        _connection.AddRequest(req12);
        _connection.Flush();
              
        _connection.addListener(Netdania.JsApi.Events.ONALERTMONITORUSERACTIVITY, function (alertTriggeredResponse) {
            try {
                var alertsIDArray = _connection.getFieldValueFromSource(alertTriggeredResponse.data.f, Netdania.JsApi.Alert.Fields.ALERT_ID);
                var notificationType = _connection.getFieldValueFromSource(alertTriggeredResponse.data.f, Netdania.JsApi.Alert.Fields.NOTIFICATION_TYPE)[0];

                /*
                If the notification is about alerts
                 */
                if (alertsIDArray != null) {
                    $.each(alertsIDArray, function (i) {
                        var alertID = alertsIDArray[i];

                        switch (notificationType) {
                            case Netdania.JsApi.Alert.NotificationTypes.ALERT_EDITED:
                            {
                                _getAlertFromServer(alertID, Netdania.JsApi.Alert.NotificationTypes.ALERT_EDITED);
                                break;
                            }
                            case Netdania.JsApi.Alert.NotificationTypes.ALERT_ADDED:
                            {
                                _getAlertFromServer(alertID, Netdania.JsApi.Alert.NotificationTypes.ALERT_ADDED);
                                break;
                            }
                            case Netdania.JsApi.Alert.NotificationTypes.ALERT_DELETED:
                            {
                                Netdania.Alert.AlertMonitor.AlertDeleted(Netdania.Alert.AlertStorage.GetAlert(alertID));
                                Netdania.Alert.AlertStorage.RemoveAlert(alertID);

                                if (globalAlertObject) {
                                    globalAlertObject._refreshDeletedAlertsGrid();
                                    globalAlertObject._refreshGrid();
                                }
                                break;
                            }

                        }
                    });
                }

                /*
                If notification is about user
                 */
                switch (notificationType) {
                    case Netdania.JsApi.Alert.NotificationTypes.PUSH_DEVICES_CHANGED:
                    {
                        Netdania.Alert.GetPushDevices(userID);
                        break;
                    }
                }

            } catch (err) {
                Netdania.JsApi.Utilities.LogError("MonitorUserAlerts.ONALERTMONITORUSERACTIVITY Error", err);
            }
        });

        _connection.addListener(Netdania.JsApi.Events.ONALERTMONITORUSER, function (alertTriggeredResponse) {
            try {
                var alertID = _connection.getFieldValueFromSource(alertTriggeredResponse.data.f, Netdania.JsApi.Alert.Fields.ALERT_ID)[0];
                var alertObject = Netdania.Alert.AlertStorage.GetAlert(alertID);
                if (alertObject != undefined) {

                    Netdania.Alert.AlertMonitor.AlertDeleted(alertObject);

                    /*
                    Save to the triggered alerts
                    */
                    alertObject.triggerDate =  new Date().getTime();
                    alertObject.triggerLevel =  _connection.getFieldValueFromSource(alertTriggeredResponse.data.f, Netdania.JsApi.Alert.Fields.TRIGGER_VALUE)[0];
                    Netdania.Alert.TriggeredAlertStorage.SaveAlert({userID: userID, alert: alertObject});

                    Netdania.Alert.AlertStorage.RemoveAlert(alertID);

                    var currentDate = new Date();
                    customPopup = new Netdania.Util.PopUp();
                    customPopup.initPopup({ containerName: 'popupContainer', cssClass: "alert-popup" });
                    var ALERT_NOTIFICATION_TITLE = "Price alert";
                    var ALERT_NOTIFICATION = "<div class='price-alert-container break-word'>Name: {0} <br /> Field: {1} <br /> Alert level: {2} <br /> Trigger level: {3} <br /> Received at: {4}  <br /> Message: {5}</div>";
                    customPopup.showPopup({ title: ALERT_NOTIFICATION_TITLE, dialogHeight: 250, message: ALERT_NOTIFICATION
                        .replace("{0}", alertObject.name)
                        .replace("{1}", alertObject.fieldName)
                        .replace("{2}", alertObject.alertValue)
                        .replace("{3}", _connection.getFieldValueFromSource(alertTriggeredResponse.data.f, Netdania.JsApi.Alert.Fields.TRIGGER_VALUE)[0])
                        .replace("{4}", currentDate.toString())
                        .replace("{5}", alertObject.message),
                        showButtons: true,
                        messageType: 'successfull',
                        okButtonText: "View FullQuote",
                        callbackFunction: function () {
                            customPopup.closePopup();
                            window.location.href = "/Products/FullQuote/FullQuote.aspx?symbol=" + alertObject.symbol + "&provider=" + alertObject.provider;
                        },
                        viewAllButton: { text: "View All", callbackFunction: function () {
                            customPopup.closePopup();
                            window.location.href = "/Alerts/AlertsCentral.aspx";
                        } },
                        cancelButtonText: "Close"
                    });

                    /*
                    In case that the applet is on the page
                    var position = $('.alert-popup').position();
                    var _width = $('.alert-popup').width();
                    var _height = $('.alert-popup').height();
                    $('.alert-popup').bgiframe1({ width: _width, height: _height, left: position.left, top: position.top, element: '.alert-popup' });
                    */
                }


                if (globalAlertObject) {
                    globalAlertObject._refreshSentAlertsGrid();
                    globalAlertObject._refreshGrid();
                }
            } catch (err) {
                Netdania.JsApi.Utilities.LogError("MonitorUserAlerts", err);
            }

        });
    }
    catch (err) {
        Netdania.JsApi.Utilities.LogError("MonitorUserAlerts.ONALERTMONITORUSER Event Error", err);
    }
}

Netdania.Alert.GetPushDevices = function(userID){
    var connectionConfig = {
        pollingInterval: Netdania.JsApi.PoolingInterval.AUTO, //milliseconds
        host: Netdania.Alert.Config.HostAlerts,
        behavior: Netdania.JsApi.ConnectionType.AUTO, //LONG POLLING
        type: 1 //the server response format: JSONP
    }
    //create a json connection object
    var _connection = new Netdania.JsApi.JSONConnection(connectionConfig);

    var pushDevicesObject = new Netdania.JsApi.Alert.AlertObject();
    pushDevicesObject.USER_ID = userID;
    pushDevicesObject.COOKIE = "NDWebsite";

    var reqPushDevices = Netdania.JsApi.Request.getReqObjAlertGetPushDevices(pushDevicesObject);
    _connection.AddRequest(reqPushDevices);
    _connection.Flush();

    _connection.addListener(Netdania.JsApi.Events.ONALERTGETPUSHDEVICES, function (getPushDevicesResponse) {
        _pushDevices = [];
        if (getPushDevicesResponse.data.f[0].v.length > 0) {
            for (var i = 0; i < getPushDevicesResponse.data.f[0].v.length; i++) {

                var deviceName = _connection.getFieldValueFromSource(getPushDevicesResponse.data.f[0].v[i].f, Netdania.JsApi.Alert.Fields.PUSH_DEVICES_NAME);
                deviceName = (deviceName != null && deviceName != undefined && deviceName.length > 0 ? deviceName[0] : '');

                var deviceID = _connection.getFieldValueFromSource(getPushDevicesResponse.data.f[0].v[i].f, Netdania.JsApi.Alert.Fields.PUSH_DEVICES_ID);
                deviceID = (deviceID != null && deviceID != undefined && deviceID.length > 0 ? deviceID[0] : '');

                var isEnabled = _connection.getFieldValueFromSource(getPushDevicesResponse.data.f[0].v[i].f, Netdania.JsApi.Alert.Fields.PUSH_DEVICE_STATE);
                isEnabled = (isEnabled != null && isEnabled != undefined && isEnabled.length > 0 ? isEnabled[0] : false);

                var pushSources = _connection.getFieldValueFromSource(getPushDevicesResponse.data.f[0].v[i].f, Netdania.JsApi.Alert.Fields.PUSH_SOURCES);
                pushSources = (pushSources != null && pushSources != undefined && pushSources.length > 0 ? pushSources[0] : '');

                var deviceType = _connection.getFieldValueFromSource(getPushDevicesResponse.data.f[0].v[i].f, Netdania.JsApi.Alert.Fields.PUSH_DEVICES_TYPE);
                deviceType = (deviceType != null && deviceType != undefined && deviceType.length > 0 ? deviceType[0] : '');

                if (isEnabled) {
                        _pushDevices.push({ v: deviceName, id: deviceID, selected: false, appType: deviceType });
                }
            }
        }

        Netdania.Alert.AlertStorage.AddAllPushDevices(_pushDevices);
        if (globalAlertObject) {
            globalAlertObject.updatePushDevices();
        }

    });
}

Netdania.Alert.AddAlert = function (symbol, provider, decimals) {
    var _user = new Netdania.Util.User();
    if (_user.isAuthenticated()) {
        /*
        Alert server supports only a limited number of alerts per user account
         */
        var alertsCount = Netdania.Alert.AlertStorage.GetAllAlertsCount();
        if (alertsCount < Netdania.Alert.Config.MaxAlerts) {

            configPopup = { containerName: 'popupContainer' };
            customPopup = new Netdania.Util.PopUp();
            customPopup.initPopup(configPopup);

            $("#nd-popupContainerDiv").append(
                '<div>' +
                    '<div id="circularGLoaderDiv" style="display: none; height:100%; width:100%;  z-index: 10000; background-color:#f8f8ff; opacity: .5; filter: alpha(opacity=50); position:fixed; top: 0px; left: 0px;">' +
                    '<img src="/JSComponents/cssgeneral/images/loading.gif" style="position:fixed; top: 50%; left: 50%;">' +
                    '</div>' +
                    '<div id="nd-alert_dialog-form"  title="">' +
                    '<div id="alertContainer" ></div>' +
                    '</div>' +
                    '</div>');

            try {
                alertComponent = new Netdania.Alert.AlertComponent.RenderAlertComponent({
                    containerId: 'alertContainer',
                    userID: _user.getUserId(),
                    daysToLive: '7',
                    emailFrom: 'no-reply-alerts@internet.netdania.com',
                    emailSubject: 'Alert!!!',
                    emailContent: "Instrument name: ${INSTRUMENT}\n Alert level: ${ALERT_LEVEL}\nTrigger level: ${CURRENT_VALUE}\nTrigger time: ${DATE}\n",
                    alertFields: '',
                    allowedAlertProviders: '',
                    dateFormat: "dd-mm-yy"
                });
                alertComponent.initRenderAlertComponent();
            }
            catch (Ex) {
                alert(Ex);
            }

            $("#nd-alert_dialog-form").dialog({
                autoOpen: false,
                height: 430,
                width: 470,
                modal: true,
                draggable: true,
                dialogClass: "alert-popup",
                position: "center"
            });

            alertComponent.initAddAlertForm({ symbol: symbol, provider: provider, decimals:decimals });
            if (alertComponent.checkSymbol(symbol + '|' + provider)) {
                $("#nd-alert_dialog-form").dialog('option', 'title', 'Add alert');
                $("#nd-alert_dialog-form").dialog("open");
            }

        } else {
            customPopup = new Netdania.Util.PopUp();
            customPopup.initPopup({ containerName: 'popupContainer', cssClass: "alert-popup" });
            customPopup.showPopup({ title: 'Add alert', message: 'The maximum alerts limit has been reached!', content: '', messageType: 'warning', noCancelButton: true,
                callbackFunction: function () {
                    customPopup.closePopup();
                }
            });
        }

    }
    else {
        customPopup = new Netdania.Util.PopUp();
        customPopup.initPopup({ containerName: 'popupContainer', cssClass: "alert-popup" });
        customPopup.showPopup({ title: 'Add alert', message: 'You have to sign in before adding alerts!', content: '', messageType: 'successfull', cancelButtonText: "Close", okButtonText: "Sign In",
            callbackFunction: function () {
                /*
                  We call login page with return parameter to the same page
                  We append to the page parameters a alert parameter that contains the symbol and provider
                 */
                var urlAlert = window.location.pathname + window.location.search;
                urlAlert = Netdania.Util.addParameter(urlAlert, Netdania.Alert.Config.AddAlertUrlParameter, symbol + "|" + provider, false );
                window.location.href = urlAlert;
                window.location.href = "/UI/UserLogin.aspx?ReturnUrl=" + urlAlert;
            }
        });

    }
}

Netdania.Alert.AlertWrapper = function (config) {
    var TOCKEN = "|";
    var _condition = undefined;
    var _instrument = undefined;
    var _self = this;
    var alertObject = new Netdania.JsApi.Alert.AlertObject();

    alertObject.ALERT_ID = config.alertID;
    alertObject.USER_ID = config.userID;
    alertObject.PREVIOUS_OWNER = config.previousOwner;
    alertObject.ALERT_CONDITION = config.alertCondition;
    alertObject.DAYS_TO_LIVE = config.daysToLive;
    alertObject.SMS_PHONE_NUMBER = config.smsPhoneNumber;
    alertObject.SMS_FROM_NAME = config.smsFromName;
    alertObject.SMS_MESSAGE = config.smsMessage;
    alertObject.EMAIL_ADDRESS = config.emailAddress;
    alertObject.EMAIL_FROM = config.emailFrom;
    alertObject.EMAIL_SUBJECT = config.emailSubject;
    alertObject.EMAIL_CONTENT = config.emailContent;
    alertObject.YAHOO_ID = config.yahooIDs;
    alertObject.YAHOO_MESSAGE = config.yahooMessage;
    alertObject.SKYPE_ID = config.skypeIDs;
    alertObject.SKYPE_MESSAGE = config.skypeMessage;
    alertObject.MSN_ID = config.msnIDs;
    alertObject.MSN_MESSAGE = config.msnMessage;
    alertObject.TRIGGER_DATE = config.triggerDate;
    alertObject.ORDER_TYPE = config.orderTypeWrapper;
    alertObject.PUSH_DEVICES = config.pushDevices;
    alertObject.PUSH_MESSAGE = config.alertMessage;
    alertObject.Condition = _condition;
    alertObject.Instrument = _instrument;
    alertObject.PUSH_DEVICES_ID = config.pushDevices;
    alertObject.DELETION_DATE = config.deletionDate;
    alertObject.DELETION_REASON = config.deletionReason;
    alertObject.TIMESCALE_SECONDS = config.timescaleSeconds;
    alertObject.TRIGGER_LEVEL = config.triggerLevel;
    alertObject.ALERT_LEVEL = config.alertLevel;


    this.setCondition = function (condition) {
        _condition = condition;
    }

    this.setInstrument = function (instrument) {
        _instrument = instrument;
    }

    this.getCondition = function () {
        return _condition;
    }

    this.getInstrument = function () {
        return _instrument;
    }
    this.getAlertObject = function () {
        return alertObject;
    }
    this.getFieldName = function () {
        for (var i = 0; i < config.alertFields; i++) {
            if (config.alertFields[i].v == alertObject.Instrument.getField()) {
                return config.alertFields[i].t;
            }
        }
        return '';
    }

    this.setEmailContent = function (intrumentName, alertValue, field) {
        alertObject.EMAIL_CONTENT = alertObject.EMAIL_CONTENT.replace("${INSTRUMENT}", intrumentName).replace("${ALERT_LEVEL}", alertValue);
    }

    this.GetStorageObject = function () {

        var messageFromMail =  (alertObject.EMAIL_CONTENT  !== undefined ? alertObject.EMAIL_CONTENT.split("Message: ")[1] === undefined ? '' : alertObject.EMAIL_CONTENT.split("Message: ")[1] : "");

        var alertObjectStorage = {id:alertObject.ALERT_ID,
            symbol:alertObject.Instrument.getSymbol(),
            provider:alertObject.Instrument.getProvider(),
            name:alertObject.Instrument.getName(),
            field:alertObject.Instrument.getField(),
            alertValue:alertObject.Condition.getAlertValue(),
            fieldName:Netdania.Alert.getFieldNameByValue(alertObject.Instrument.getField()),
            message:messageFromMail,
            operator:alertObject.Condition.getRelationalOperator(),
            triggerDate: alertObject.TRIGGER_DATE,
            triggerLevel: alertObject.TRIGGER_LEVEL,
            alertLevel: alertObject.ALERT_LEVEL,
            emailContent: alertObject.EMAIL_CONTENT,
            type: alertObject.Instrument.getType()};

          return alertObjectStorage;

    }


    this.setConditionToString = function () {
        var buffer = '';
        //alert.ALERT_CONDITION = "SHOW|NT|1352121485708.914062|I|10|GBPUSD|GBP (USD)|netdania_rt|>|21680.774000|TO|365|";

        if (_condition.getTimestamp() != -1) {
            buffer += "SHOW" + TOCKEN + "NT" + TOCKEN + _condition.getTimestamp() + TOCKEN;
        }

        buffer += 'I' + TOCKEN + _instrument.getField() + TOCKEN + _instrument.getSymbol() + TOCKEN + _instrument.getName() + TOCKEN + _instrument.getProvider() + TOCKEN;

        buffer += _condition.getRelationalOperator().v + TOCKEN + _condition.getAlertValue() + TOCKEN;

        if (_condition.getTimeout() != -1) {
            buffer += "TO" + TOCKEN + _condition.getTimeout() + TOCKEN;
        }

        if (_condition.getCustomers() != undefined && _condition.getCustomers().length > 0) {
            buffer += "CID" + TOCKEN + _condition.getCustomers() + TOCKEN;
        }

        alertObject.ALERT_CONDITION = buffer.toString();
    }

    this.setConditionToObject = function (condition) {
        var conditionArray = condition.split(TOCKEN);

        var tokenIndex = 0;
        var condition = new Netdania.Alert.SimpleConditionWrapper();

        if (conditionArray[tokenIndex] == "SHOW") {
            tokenIndex++;
        }

        if (conditionArray[tokenIndex] == "NT") {
            tokenIndex++;
            var timestamp = conditionArray[tokenIndex++];
            condition.setTimestamp(timestamp);
        }

        if (conditionArray[tokenIndex] == "I") {
            //line
            tokenIndex++;
            var instrument = new Netdania.Alert.AlertInstrument();
            var field = conditionArray[tokenIndex++];
            instrument.setField(field);
            var symbol = conditionArray[tokenIndex++];
            instrument.setSymbol(symbol);

            var name = conditionArray[tokenIndex++];
            instrument.setName(name);

            var provider = conditionArray[tokenIndex++];
            instrument.setProvider(provider);

            var operator = conditionArray[tokenIndex++];
            condition.setRelationalOperator(operator);

            var alertValue = conditionArray[tokenIndex++];
            condition.setAlertValue(alertValue);

            if (operator == "T") {
                instrument.setType(Netdania.Alert.Types.TRENDLINE);

                operator = "is"
                if (alertValue === "A") {
                    alertValue = "Above";
                }
                else if (alertValue === "B") {
                    alertValue = "Below";
                }

                condition.setAlertValue(alertValue);
                condition.setRelationalOperator(operator);

                var slope = conditionArray[tokenIndex++];
                condition.setSlope(slope);

                var initialX = conditionArray[tokenIndex++];
                condition.setInitialX(initialX);

                var initialY = conditionArray[tokenIndex++];
                condition.setInitialY(initialY);

                var timescale = conditionArray[tokenIndex++];
                condition.setTimescale(timescale);



            } else {
                instrument.setType(Netdania.Alert.Types.PRICE);
            }

        }

        if (tokenIndex < conditionArray.length && conditionArray[tokenIndex] == ("TO")) {
            tokenIndex++;
            var timeout = conditionArray[tokenIndex++];
            condition.setTimeout(timeout);
        }

        if (tokenIndex < conditionArray.length && conditionArray[tokenIndex] == ("CID")) {
            tokenIndex++;
            var customers = conditionArray[tokenIndex++];
            condition.setCustomers(customers);
        }

        alertObject.Condition = condition;
        alertObject.Instrument = instrument;


    }
}

Netdania.Alert.AlertInstrument = function (symbol, provider, name, type) {

    /**
     * The field from where to gets the value to compare with.
     */
    var _field = -1;

    /**
     * The symbol of the instrument.
     */
    var _symbol = undefined;

    /**
     * The instrument's provider.
     */
    var _provider = undefined;

    /**
     * The name of the instrument.
     */
    var _name = undefined;

    _symbol = symbol;
    _provider = provider;
    if (name != null) {
        var index = name.indexOf('|');
        if (index > 0) {
            name = name.substring(0, index - 1);
        }
    }
    _name = name;

    var _type = type;

    /**
     * Gets the instrument field.
     * @return    the field.
     */
    this.getField = function () {
        return _field;
    }

    /**
     * Sets the new field for this instrument to compare with it.
     * @param field    the new field.
     */
    this.setField = function (field) {
        _field = field;
    }

    /**
     * Gets the symbol of the instrument.
     * @return    the symbol of the instrument.
     */
    this.getSymbol = function () {
        return _symbol;
    }

    /**
     * Sets the new symbol of this instrument.
     * @param symbol    the new symbol.
     */
    this.setSymbol = function (symbol) {
        _symbol = symbol;
    }

    /**
     * Gets the instrument provider.
     * @return    the instrument provider.
     */
    this.getProvider = function () {
        return _provider;
    }

    /**
     * Sets the provider.
     * @param provider    the new provider.
     */
    this.setProvider = function (provider) {
        _provider = provider;
    }

    /**
     * Gets the name of the instrument.
     * @return the name of the instrument.
     */
    this.getName = function () {
        return _name == null ? _symbol : _name;
    }

    /**
     * Sets the name of the instrument.
     * @param name the name to set.
     */
    this.setName = function (name) {
        _name = name;
    }

    /**
     * Gets the instrument field.
     * @return    the field.
     */
    this.getType = function () {
        return _type;
    }

    /**
     * Sets the new field for this instrument to compare with it.
     * @param field    the new field.
     */
    this.setType = function (type) {
        _type = type;
    }


    this.toString = function () {
        return "" + _field + "|" + _name + "|" + _symbol + "|" + _provider;
    }
}

Netdania.Alert.Types = {
    TRENDLINE: 'Trend-line',
    PRICE: 'Horizontal-line'
}

Netdania.Alert.SimpleConditionWrapper = function () {

    /**
     * The time of the last transaction when the condition was created.
     */
    var _timestamp = -1;

    /**
     * The time when the condition expire.
     */
    var _timeout = -1;

    var _relationalOperator = ">";

    var _valueAlert = 0;

    /**
     * The condition.
     */
    var _conditionWrapper = undefined;

    var _customers = [];

    var _timescale;

    var _slope

    var _initialX;

    var _initialY;

    /**
     * Create a simple condition.
     */
    this.SimpleConditionWrapper = function () {
        this.setTimestamp(new Date().getTime());
    }

    /**
     * Gets the time of the condition.
     * @return    the time when the condition was created.
     */
    this.getTimestamp = function () {
        return _timestamp;
    }

    /**
     * Sets the time of the condition.
     * @param timestamp    the time when the condition was created.
     */
    this.setTimestamp = function (timestamp) {
        _timestamp = timestamp;
    }

    /**
     * Gets the time when the condition expire.
     * @return    the time when the condition expire.
     */
    this.getTimeout = function () {
        return _timeout;
    }

    /**
     * Sets the time when the condition expire.
     * @param timeout    the time when the condition expire.
     */
    this.setTimeout = function (timeout) {
        _timeout = timeout;
    }

    /**
     * Gets the operator to be applied.
     * @return    the operator used to comparison.
     */
    this.getRelationalOperator = function () {
        return _relationalOperator;
    }

    /**
     * Sets the operator to be applied.
     * @param relationalOperator    the operator applied.
     */
    this.setRelationalOperator = function (relationalOperator) {
        _relationalOperator = relationalOperator;
    }

    /**
     * Gets the value to compare.
     * @return    the value used to comparison.
     */
    this.getAlertValue = function () {
        var value = parseFloat(_valueAlert);
        if (isNaN(value)) {
            value = _valueAlert;
        }
        return value;
    }

    this.getTimeScale = function () {
        var value = _timescale;
        if (isNaN(value)) {
            value = "";
        }
        return value;
    }

    /**
     * Sets the value to compare.
     * @param value    the operator applied.
     */
    this.setAlertValue = function (valueAlert) {
        _valueAlert = valueAlert;
    }

    /**
     * Gets the condition.
     * @return    the condition.
     */
    this.getConditionWrapper = function () {
        return _conditionWrapper;
    }

    /**
     * Sets the condition.
     * @param conditionWrapper    the condition.
     */
    this.setConditionWrapper = function (conditionWrapper) {
        _conditionWrapper = conditionWrapper;
    }

    this.getCustomers = function () {
        return _customers;
    }

    this.setCustomers = function (customers) {
        _customers = customers;
    }

    this.setSlope = function (slope) {
        _slope = slope;
    }

    this.setInitialX = function (initialX) {
        _initialX = initialX;
    }

    this.setInitialY = function (initialY) {
        _initialY = initialY;
    }

    this.setTimescale = function (timescale) {
        _timescale = timescale;
    }
}

Netdania.Alert.getFieldNameByValue = function (value) {
    var alertFieldsNames = [
        { v: "10", t: "Bid" }
        ,
        { v: "11", t: "Ask" }
        ,
        { v: "6", t: "Close"}
    ];
    for (var i = 0; i < alertFieldsNames.length; i++) {
        if (alertFieldsNames[i].v == value) {
            return alertFieldsNames[i].t;
        }
    }
    return '';
}

Netdania.Alert.DefaultContactStorage = (function () {
    var _storrageName = "defaultContacts";
    var _contactsObject = {};

    GetContactsFromStorage = function () {
        var allContactsObject = {};
        try {
            var alertsString = $.ezstorage.get(_storrageName);
            if (alertsString !== undefined) {
                if (typeof alertsString == "string") {
                    allContactsObject = JSON.parse(alertsString);
                    if (allContactsObject !== undefined && typeof allContactsObject == "string") {
                        allContactsObject = JSON.parse(allContactsObject);
                    }
                }
                else {
                    allContactsObject = alertsString;
                }
            }

        } catch (err) {
            Netdania.JsApi.Utilities.LogError("ContactStorage.GetContactsFromStorage error", err);
        }
        return  allContactsObject;
    }
    GetUserContactsFromStorage = function () {

        try {
            var allContactsObject = GetContactsFromStorage();

            if (allContactsObject.users !== undefined) {
                var _user = new Netdania.Util.User();
                var id = _user.getUserId();
                var user = $.grep(allContactsObject.users, function (e) {
                    return e.id == id;
                });

                if (user[0] !== undefined) {
                    _contactsObject = user[0];
                }

            }

        } catch (err) {
            Netdania.JsApi.Utilities.LogError("ContactStorage.GetUserContactsFromStorage error", err);
        }
        return  _contactsObject;
    }
    SetContactsToStorage = function (contacts) {
        try {
            $.ezstorage.set(_storrageName, contacts, {persist: true});
        } catch
            (err) {
            Netdania.JsApi.Utilities.LogError("ContactStorage.SetContactsToStorage error", err);
        }
    }
    SetUserContactsToStorage = function () {
        try {
            var allContactsObject = GetContactsFromStorage();

            var _user = new Netdania.Util.User();
            var id = _user.getUserId();
            if (allContactsObject.users !== undefined){
                var found = false;
                $.each(allContactsObject.users, function (i) {
                    if (allContactsObject.users[i].id == id) {
                        allContactsObject.users[i] = _contactsObject;
                        found = true;
                        return false;
                    }
                });

                if (!found){
                    _contactsObject.id = id;
                    allContactsObject.users.push(_contactsObject);
                }
            }
            else
            {
                _contactsObject.id = id;
                allContactsObject.users = [_contactsObject];
            }

            SetContactsToStorage(allContactsObject);
        } catch
            (err) {
            Netdania.JsApi.Utilities.LogError("ContactStorage.SetUserContactsToStorage error", err);
        }
    }

    return {
        AddContact: function (contact) {
            try {
                GetUserContactsFromStorage();
                if (_contactsObject.contacts !== undefined) {
                    var found = false;
                    $.each(_contactsObject.contacts, function (i) {
                        if (_contactsObject.contacts[i].v == contact.v && _contactsObject.contacts[i].v == contact.t) {
                            found = true;
                            return false;
                        }
                    });

                    if (!found) {
                        _contactsObject.contacts.push(contact);
                    }
                }
                else {
                    _contactsObject.contacts = [contact];
                }
                SetUserContactsToStorage();

            } catch
                (err) {
                Netdania.JsApi.Utilities.LogError("DefaultContactStorage.AddContact Error", err);
            }
        },
        RemoveContact: function (contact) {
            try {
                GetUserContactsFromStorage();
                if (_contactsObject.contacts !== undefined) {
                    $.each(_contactsObject.contacts, function (j) {
                        if (_contactsObject.contacts[j].v == contact.v && _contactsObject.contacts[j].t == contact.t) {
                            _contactsObject.contacts.splice(j, 1);
                            SetUserContactsToStorage();
                        }
                    });
                }

            } catch (err) {
                Netdania.JsApi.Utilities.LogError("DefaultContactStorage.RemoveContact Error", err);
            }
        },
        /**
         *
         * @param filter t for type
         * @returns {Array} all default contacs filterd by type
         * @constructor
         */
        GetContacts: function (filter) {
            var contacts = [];
            try {
                GetUserContactsFromStorage();
                if (_contactsObject.contacts !== undefined) {
                    contacts = $.grep(_contactsObject.contacts, function (e) {
                        return (filter.t !== undefined && e.t == filter.t);
                    });
                }
            } catch (err) {
                Netdania.JsApi.Utilities.LogError("DefaultContactStorage.GetContacts error", err);
            }
            return contacts;
        }
    }

})();

Netdania.Alert.AlertStorage = (function () {
    var _storrageName = "monitoredAlerts";
    var _cookieSyncName = "monitoredAlertsHash";
    var _alertsObject = {};
    var _contactsList = [];

    var GetAlertsFromStorage = function () {
        var alertsObject = {};
        try {
            alertsString = $.ezstorage.get(_storrageName);
            if (alertsString !== undefined) {
                if (typeof alertsString == "string") {
                    alertsObject = JSON.parse(alertsString);
                    if (alertsObject !== undefined && typeof alertsObject == "string") {
                        alertsObject = JSON.parse(alertsObject);
                    }
                }
                else {
                    alertsObject = alertsString;
                }
            }
        } catch (err) {
            Netdania.JsApi.Utilities.LogError("AlertStorage.GetAlertsFromStorage error", err);
        }
        return  alertsObject;
    }

    var SetAlertsToStorage = function () {
        try {
            $.ezstorage.set(_storrageName, _alertsObject);
            var alertsString = $.ezstorage.get(_storrageName);
            var alertsHash = Netdania.Util.HashCode(JSON.stringify(alertsString));
            var cookies = Netdania.Util.Cookies()
            cookies.setCookie(_cookieSyncName, alertsHash);
        } catch
            (err) {
            Netdania.JsApi.Utilities.LogError("AlertStorage.SetAlertsToStorage error", err);
        }
    }

    var DeleteAlertsFromStorage =  function() {
        try {
            $.ezstorage.remove(_storrageName);
            var alertsHash = Netdania.Util.HashCode("");
            var cookies = Netdania.Util.Cookies()
            cookies.setCookie(_cookieSyncName, alertsHash);
        } catch
            (err) {
            Netdania.JsApi.Utilities.LogError("MonitorUserAlerts.DeleteAlertsFromStorage error", err);
        }
    }

    return {
        ExistAlerts: function () {
            var cookies = Netdania.Util.Cookies()
            //check cookie for http/https scope difference
            var cookieHash = cookies.getCookie(_cookieSyncName);
            if (cookieHash !== undefined) {
                var alertsString = $.ezstorage.get(_storrageName);
                if (alertsString === undefined || alertsString == null || !alertsString || alertsString == '') {
                    return false;
                }
                else {

                    var alertsObject = GetAlertsFromStorage();
                    if (alertsObject == null || alertsObject == undefined || alertsObject.items == null || alertsObject.items == undefined){
                        return false;
                    }

                    var alertsHash = Netdania.Util.HashCode(JSON.stringify(alertsString));
                    if (alertsHash == cookieHash) {
                        return true;
                    }
                    else {
                        return false;
                    }
                }
            }
            else {
                return false;
            }
        },
        AddAllAlerts: function (alerts) {
            try {
                _alertsObject = _alertsObject || {};
                _alertsObject.items = [];
                $.each(alerts, function (i) {
                    _alertsObject.items.push(alerts[i]);
                });
            } catch
                (err) {
                Netdania.JsApi.Utilities.LogError("AlertStorage.AddAllAlerts error", err);
            }
            SetAlertsToStorage();
        },
        GetAlert: function(alertID) {
            _alertsObject = GetAlertsFromStorage();
            var alert = undefined;
            try {
                $.each(_alertsObject.items, function (i) {
                    if (_alertsObject.items[i].id == alertID) {
                        alert = _alertsObject.items[i];
                        return false;
                    }
                });
            } catch
                (err) {
                Netdania.JsApi.Utilities.LogError("AlertStorage.GetAlert error", err);
            }
            return alert;
        },
        GetAllAlerts: function (){
            _alertsObject = GetAlertsFromStorage();
            if (_alertsObject != undefined && _alertsObject.items != undefined){
                return _alertsObject.items;
            }
            return undefined;

        },
        GetAllAlertsCount: function (){
            var count = 0;
            var alerts = this.GetAllAlerts();
            if (alerts != null && alerts != undefined){
                count = alerts.length;
            }
            return count;
        },
        SaveAlert: function(alertObject){
            _alertsObject = GetAlertsFromStorage();
            var boolSave = true;
            $.each(_alertsObject.items, function (i) {
                if (_alertsObject.items[i].id == alertObject.id) {
                    boolSave = false;
                    return false;
                }
            });

            if (boolSave) {
                _alertsObject.items.push(alertObject);
                SetAlertsToStorage();
            }

        },
        ReplaceAlert: function (alertObject) {
            _alertsObject = GetAlertsFromStorage();
            try {
                $.each(_alertsObject.items, function (i) {
                    if (_alertsObject.items[i].id == alertObject.id) {
                        _alertsObject.items[i] = alertObject;
                        SetAlertsToStorage();
                        return false;
                    }
                });
            } catch
                (err) {
                Netdania.JsApi.Utilities.LogError("AlertStorage.Replace error", err);
            }
        },
        RemoveAlert: function (alertID) {
            _alertsObject = GetAlertsFromStorage();
            try {
                if (_alertsObject.items !== undefined) {
                $.each(_alertsObject.items, function (i) {
                    if (_alertsObject.items[i].id == alertID) {
                        _alertsObject.items.splice(i, 1);
                        SetAlertsToStorage();
                        return false;
                    }
                });
               }
            } catch
                (err) {
                Netdania.JsApi.Utilities.LogError("AlertStorage.RemoveAlert error", err);
            }
        },
        DeleteAlerts: function () {
            DeleteAlertsFromStorage();
        },

        AddAllPushDevices: function (pushDevices) {
            try {
                _alertsObject = GetAlertsFromStorage();
                _alertsObject.pushdevices = [];
                $.each(pushDevices, function (i) {
                    _alertsObject.pushdevices.push(pushDevices[i]);
                });
            } catch
                (err) {
                Netdania.JsApi.Utilities.LogError("AlertStorage.AddPushDevice error", err);
            }
            SetAlertsToStorage();
        },
        GetAllPushDevices: function (){
            _alertsObject = GetAlertsFromStorage();
            if (_alertsObject ==null || _alertsObject == undefined ||_alertsObject.pushdevices == null || _alertsObject.pushdevices == undefined){
                return [];
            }
            return _alertsObject.pushdevices;
        },
        ContactsList: _contactsList
    }

})();

Netdania.Alert.AlertMonitor = (function () {

    var imageAlertNotSet = '/Alerts/js/images/bell.png';
    var imageAlertSet = '/Alerts/js/images/bell-set.png';
    var _alertsImages = [];

    RemoveAlert = function (context) {
        context.image.alerts = context.image.alerts || [];
        $.each(context.image.alerts, function (i) {
            if (context.image.alerts[i].id == context.alert.id) {
                context.image.alerts.splice(i, 1);
            }
        });

        if (context.image.alerts.length == 0) {
            $('#' + context.image.id).attr("src", imageAlertNotSet);
            $('#' + context.image.id).attr("title", "Add " + context.alert.name + " alert.");
        }
        else
        {
            $('#' + context.image.id).attr("title", "Add " + context.alert.name + " alert. Active alerts: " + context.image.alerts.length);
        }

    }
    AddAlert = function (context) {
        try {
            context.image.alerts = context.image.alerts || [];
            context.image.alerts.push({id: context.alert.id, fieldName: context.alert.fieldName, alertValue: context.alert.alertValue, name: context.alert.name});
            $('#' + context.image.id).attr("src", imageAlertSet);
            $('#' + context.image.id).attr("title", "Add " + context.alert.name + " alert. Active alerts: " + context.image.alerts.length);
        }
        catch (err) {
            Netdania.JsApi.Utilities.LogError("AlertMonitor.AddAlert", err);
        }

    }

    return {
        AlertImage: function (options) {
            var symbol = options.s;
            var provider = options.p;
            var name = options.n || symbol;
            var decimals = options.d;

            var imageHtml = '';
            try {
                var id = _alertsImages.length;
                var found = false;
                $.each(_alertsImages, function (i) {
                    if (_alertsImages[i].symbol == symbol && _alertsImages[i].provider == provider) {
                        found = true;
                        id = i;
                        return false;
                    }
                });

                var idString = "alert-image-" + id;

                if (!found) {
                    _alertsImages.push({symbol: symbol, provider: provider, id: idString, decimals:decimals});

                }
                imageHtml = '<img onclick="Netdania.Alert.AlertMonitor.AlertImageClick(this, event)" style="cursor:pointer;vertical-align:middle" class="nd-alert-image" id="' + idString + '" title="Add ' + name + ' alert." src="' + imageAlertNotSet + '"/>';

            }
            catch (err) {
                Netdania.JsApi.Utilities.LogError("AlertMonitor.AlertImage", err);
            }
            return imageHtml;

        },
        AlertAdded: function (AddedAlert) {
            if (AddedAlert)
            {
                $.each(_alertsImages, function (i) {
                    if (_alertsImages[i].symbol == AddedAlert.symbol && _alertsImages[i].provider == AddedAlert.provider) {
                        AddAlert({image: _alertsImages[i], alert: AddedAlert});
                    }
                });
            }

        },
        AlertDeleted: function (DeletedAlert) {
            try {
                if (DeletedAlert && DeletedAlert !== undefined)
                {
                    $.each(_alertsImages, function (i) {
                        if (_alertsImages[i].symbol == DeletedAlert.symbol && _alertsImages[i].provider == DeletedAlert.provider) {
                            RemoveAlert({image: _alertsImages[i], alert: DeletedAlert});
                        }
                    });
                }
            }
            catch (err) {
                Netdania.JsApi.Utilities.LogError("AlertMonitor.DeletedAlert", err);
            }
        },
        SetActiveAlerts: function (cookieAlerts) {
            try {
                if (cookieAlerts != null && cookieAlerts != undefined) {
                    $.each(cookieAlerts, function (i) {
                        $.each(_alertsImages, function (j) {
                            if (_alertsImages[j].symbol == cookieAlerts[i].symbol && _alertsImages[j].provider == cookieAlerts[i].provider) {
                                AddAlert({image: _alertsImages[j], alert: cookieAlerts[i]});
                            }
                        });
                    });
                }
            }
            catch (err) {
                Netdania.JsApi.Utilities.LogError("AlertMonitor.SetActiveAlerts", err);
            }
        },
        AlertImageClick: function (image, event) {
            try {
                if (event && event.stopPropagation) {
                    event.stopPropagation();
                }
                else {
                    event.returnValue = false;
                }


                var alertImage = $.grep(_alertsImages, function (a) {
                    return a.id == image.id;
                })[0];

                if (Netdania.Alert !== undefined) {
                    Netdania.Alert.AddAlert(alertImage.symbol, alertImage.provider, alertImage.decimals);
                }
                else {
                    alert('Coming soon!');
                }

            }
            catch (err) {
                Netdania.JsApi.Utilities.LogError("AlertMonitor.AlertImageClick", err);
            }
        },
        /*
         If user is authenticated, monitor alerts
         */
        AlertControl: function () {
            try {
                var _user = new Netdania.Util.User();
                if (_user.isAuthenticated()) {
                    if (Netdania.Alert.AlertStorage.ExistAlerts()) {
                        Netdania.Alert.MonitorUserAlerts(_user.getUserId());
                    }
                    else {
                        Netdania.Alert.GetAllAlerts(_user.getUserId());
                    }
                     /*
                     If we find the alert parameter we show the add alert window
                      */
                    if (queryString(Netdania.Alert.Config.AddAlertUrlParameter) !== 'false') {
                        var parameter = queryString(Netdania.Alert.Config.AddAlertUrlParameter);
                        if (parameter.indexOf("|") !=-1) {
                            var parameters = parameter.split("|");
                            var symbol = parameters[0];
                            var provider = parameters[1];
                            Netdania.Alert.AddAlert(symbol, provider);
                        }
                    }
                }
                else {
                    Netdania.Alert.AlertStorage.DeleteAlerts();
                }
            } catch (err) {
                Netdania.JsApi.Utilities.LogError("AlertMonitor.AlertControl", err);
            }
        }
    }
})();

