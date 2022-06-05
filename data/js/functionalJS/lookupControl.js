var lookUpComponent = function (config) {
    dataCollection = [];
    Events.enable.call(this);
    var lookupName = config.inputName;
    var parent = document.getElementById(config.parentName);
    parent.className = "nd-lookup";
    var showButton = config.showButton;
    var callback = config.callback;
    var noItemFound = config.noItemFound;
    var isCombobox = config.isCombobox;
    var converterObject = config.ConverterObject;
    var typeSearch = config.typeSearch;
    var defaultValue = config.defaultValue != "" ? config.defaultValue : "Select";
    var icon = config.showIcon ? "<span class=\"lookupSearchIcon\" style=\"width:17px; float:left\">&nbsp;</span>" : "";
    var inputControl = icon + "<span class='ui-combobox'><input id=\"" + lookupName + "\" class='unfilledLookup' onfocus=\"Focused(this);\" onblur=\"LostFocus(this);\"  value =\"" + defaultValue + "\" >{0}</span>";

    if (isCombobox) {
        inputControl = inputControl.replace('{0}', '<a tabIndex="-1" id="ddBtn' + lookupName + '" title="Show All Items"  class="ui-button ui-widget ui-state-default ui-button-icon-only ui-combobox-toggle ui-icon ui-icon-triangle-1-s" role="button" aria-disabled="false"></a>');
    }
    else {
        inputControl = inputControl.replace('{0}', '');
    }
    if (config.showButton) {
        inputControl += "<span id=\"" + lookupName + "Loader\" class=\"lookupLoaderVisible\"  style=\"width:17px;\">&nbsp;</span><input id=\"btn" + lookupName + "\" onclick='Search(" + callback + ",\"" + lookupName + "\",\"\"," + noItemFound + ");'   type='button' class='show' value='Search' />";
    }
    else {
        inputControl += "<span id=\"" + lookupName + "Loader\" class=\"lookupLoaderVisible\"  style=\"width:27px;\">&nbsp;</span>";
    }
    parent.innerHTML = inputControl;


    var data = getLookupValues(converterObject.pairs);
    writeToDiv(lookupName, showButton, noItemFound, callback, converterObject, data, typeSearch, false);
    HideLoader(lookupName);

}

var dataCollection = [];


function ShowDropDownControl(lookup, closeIfOpened) {
    var input = $("#" + lookup);
    if (closeIfOpened && input.autocomplete("widget").is(":visible")) {
        input.autocomplete("close");
        return;
    }
    input.autocomplete("search", "");

}
function ShowLoader(inputName) {
    $("#" + inputName + "Loader")[0].className = "lookupLoaderVisible";
}

function HideLoader(inputName) {
    $("#" + inputName + "Loader")[0].className = "lookupLoaderHide";
}

function LostFocus(obj) {
    if (obj.value == '') {
        obj.value = 'Select';
        obj.className = 'unfilledLookup'
    }
}

function Focused(obj) {
    if (obj.value == 'Select') {
        obj.value = '';
        obj.className = 'filledLookup';
    }
}

function writeToDiv(inputName, showButton, noItemFound, callback, converterObject, data, typeSearch, showItemOnSelect) {
    $(function () {

        if (!showButton) {
            var input =  $("#" + inputName);
            input.autocomplete({
                minLength: 0,
                autoFocus:true,
                source: function (request, response) {
                    ShowLoader(inputName);
                    setTimeout(function () {
                        var results = $.ui.autocomplete.filter(data, request.term);
                        if (!results.length) {
                            results = [noItemFound];
                            lookUpComponent.disable = true;
                        }
                        response(results);
                    }, 100);
                },
                response: function (event, ui) {
                    if (ui.content.length > 0) {
                       var firstItem = ui.content[0];
                       input.data('value',firstItem);
                    }
                },
                focus: function (event, ui) {
                    event.stopPropagation();
                    if (ui.item != undefined && ui.item.label != noItemFound) {
                        if (showItemOnSelect == true) {
                            input.val(ui.item.label);

                       } else
                        {
                             if (input.val() == typeSearch){
                                 input.val(typeSearch);
                                 input.select();
                             }
                        }
                    }
                },
                select: function (event, ui) {
                    if (ui.item != undefined && ui.item.label != noItemFound) {
                        callback(ui.item);
                        input[0].value = ui.item.value;

                        input.data('value',ui.item);
                        input.data('oldvalue',ui.item);
                        input[0].className = 'filledLookup';
                        return false;
                    }  else {
                        var oldValue = input.data('oldvalue');
                        if (oldValue !== undefined && oldValue != noItemFound){
                            input[0].value = oldValue.value;
                            input.data('value',oldValue);
                            input[0].className = 'filledLookup';
                            return false;
                        }
                    }
                },
                search: function (event, ui) {
                    if (input.val() == input.data('value').value) {
                        input.val(typeSearch);
                        input.select();
                    }
                    ShowLoader(inputName);
                },
                open: function (event, ui) {
                    HideLoader(inputName);


                },
                close: function (event, ui) {
                    if (input.data('value') !== undefined){
                        if (input.data('value') != input[0].value) {
                            input[0].value = input.data('value').value;

                        }
                    }
                }
            });

            input.bind("click", {
                input:input
            }, function (event) {
                try {
                    event.data.input.val(typeSearch);
                    event.data.input.select();
                    ShowDropDownControl(inputName, false);
                }
                catch (err) {
                    Netdania.Util.LogError("writeToDiv.click", err);
                }
            });

            input.bind("keydown", {
                input:input
            }, function (event) {
                try {
                    var code = event.keyCode || event.which;
                    if(code == 13 || code == 9) { //Enter or Tab keycode
                        //select first item from the list
                       // var widget = event.data.input.autocomplete( "widget" );
                        //if (widget !== undefined && widget.children !== undefined) {
                        var selectedItem = input.data('value');
                            if ( selectedItem!== undefined && selectedItem.label != noItemFound){
                                callback(selectedItem);
                                input[0].value = selectedItem.value;
                                input[0].className = 'filledLookup';
                                input.autocomplete( "close" );
                            }
                        else{
                                selectedItem = input.data('oldvalue');
                                if (selectedItem !== undefined && selectedItem.label != noItemFound) {
                                    callback(selectedItem);
                                    input.data('value',selectedItem);
                                    input[0].value = selectedItem.value;
                                    input[0].className = 'filledLookup';
                                    input.autocomplete( "close" );
                                }
                            }
                            /*
                            var data = widget.children(0).data("item.autocomplete");

                               if (data !== undefined && data.label != noItemFound) {
                                   callback(data);
                                   input[0].value = data.value;

                                   input[0].className = 'filledLookup';
                                   input.data('value',input[0].value);
                                   input.autocomplete( "close" );
                               }  else {
                                   if (input.data('value') !== undefined){
                                       input[0].value = input.data('value');

                                       input[0].className = 'filledLookup';
                                       input.autocomplete( "close" );
                                   }
                               }
                               */
                       // }
                    }
                }
                catch (err) {
                    Netdania.Util.LogError("writeToDiv.keyup", err);
                }
            });

        }

    });
}

var getLookupValues = function (data) {
    var values = [];
    if (data != undefined) {
        for (i = 0; i < data.length; i++) {
            if (!HasValue(values, data[i].s)) {
                values.push({ label: data[i].s + ' (' +  data[i].n + ')', value: data[i].s});
            }
        }
    }

    return values;
}

function HasValue(objArray, value) {
    var contains = false;
    for (var i = 0; i < objArray.length; i++) {
        if (objArray[i].value == value) {
            contains = true;
            break;
        }
    }
    return contains;
}

var setLookupControl = function (config) {

    var parentName = config.parentName;
    var inputName = config.inputName;
    var width = config.width;
    var showButton = config.showButton;
    var callback = config.callback;
    var noItemFound = config.noItemFound;
    var lookupComponentConfig = config;

    var lookupComponent = new lookUpComponent(lookupComponentConfig);

    if (config.isCombobox) {
        $("#ddBtn" + inputName + "").click(function (event) {
            ShowDropDownControl(inputName, true);
        });
    }

    this.Select = function (ui) {
        if (ui.item != undefined && ui.item.label != noItemFound) {
            callback(ui.item);
            $("#" + inputName + "")[0].value = ui.item.label;
            $("#" + inputName + "")[0].className = 'filledLookup';
            $("#" + inputName + "").data('value',ui.item);
            $("#" + inputName + "").data('oldvalue',ui.item);
            HideLoader(inputName);
            return false;
        }
        else {
            $("#" + inputName + "")[0].value = lastInserted;
            $("#" + inputName + "")[0].className = 'unfilledLookup';
            $("#" + inputName + "").data('value',ui.item);
            $("#" + inputName + "").data('oldvalue',ui.item);
            HideLoader(inputName);
            return false;
        }
    }
    return this;
};

