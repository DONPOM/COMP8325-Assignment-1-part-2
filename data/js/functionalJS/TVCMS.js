var HTML5_Placeholder = false;
var HTML5_Autofocus = false;
var HTML5_tel = false;
var HTML5_url = false;


// check for html5 support
$().ready(function () {
    var i = document.createElement('input');
    HTML5_Placeholder = 'placeholder' in i;
    HTML5_Autofocus = 'autofocus' in i;

    i.setAttribute('type', 'tel');
    HTML5_tel = i.type != 'text';
    i.setAttribute('type', 'url');
    HTML5_url = i.type != 'text';

    //alert(HTML5_Placeholder + ' ' + HTML5_Autofocus + ' ' + HTML5_tel + ' ' + HTML5_url);

    // ASP wont let us set the type, it complains
    // set we set xtype and take care of in in JS
    $elms = $('input, textarea, select');
    $elms.each(function () {
        var xtype = $(this).attr('xtype');
        if (xtype) {
            if (xtype == 'tel' && HTML5_tel) {
                $(this)[0].setAttribute('type', 'tel'); // Work around jQuery issues
            }
            if (xtype == 'url' && HTML5_url) {
                $(this)[0].setAttribute('type', 'url'); // Work around jQuery issues
            }
            if (xtype == 'date') {
                $(this).css("text-align", "center");
                $(this).datepicker();
                var attr = $(this).attr('placeholder');
                if (typeof attr == 'undefined' || attr == false) {
                    $(this).attr('placeholder', 'DD/MM/YYYY');
                    $(this).css('text-align', 'center');
                }
            }
            if (xtype == 'money') {
                $(this).keypress(InputMoneyOnly);
                if (!$(this).hasAttr('placeholder')) {
                    $(this).attr('placeholder', '$###.##');
                }
            }
            if (xtype == 'number') {
                $(this).keypress(InputNumberOnly);
                if (!$(this).hasAttr('placeholder')) {
                    $(this).attr('placeholder', '#####.#####');
                }
            }
            if (xtype == 'digits') {
                $(this).keypress(InputDigitsOnly);
                if (!$(this).hasAttr('placeholder')) {
                    $(this).attr('placeholder', '#####');
                }
            }
            if (xtype == 'tel') {
                if (!$(this).hasAttr('placeholder')) {
                    $(this).attr('placeholder', '(###) ###-####');
                }
                $(this).keypress(InputTelOnly);
                $(this).blur(InputTelFormat);
            }
            if (xtype == 'slug') {
                $(this).keypress(InputSlugOnly);
                $(this).blur(InputSlugOnly);
            }
            if (xtype == 'email') {
                if (!$(this).hasAttr('placeholder')) {
                    $(this).attr('placeholder', 'user@isp.com');
                }
            }
            if (xtype == 'url') {
                if (!$(this).hasAttr('placeholder')) {
                    $(this).attr('placeholder', 'http://domain.com');
                }
            }
            if (xtype == 'time') {
                $(this).css("text-align", "center");
                $(this).clockpick({
                    starthour: 4,
                    endhour: 19
                });
            }
        }

        // Add required mark
        if ($(this).hasAttr('required')) {
            var reqVal = $(this).attr('reqmark');
            if (reqVal && reqVal != '')
                $('<span>').html(reqVal).addClass('FormRequired').attr('title', 'Required Field').insertAfter(this);
            else
                $('<span>').html('&lowast;').addClass('FormRequired').attr('title', 'Required Field').insertAfter(this);
        } else {
            if ($(this).width() > 0) {
                $('<span>').html('&nbsp;').addClass('FormNotRequired').insertAfter(this);
            }
        }
    });

    // Install select handler
    $('select').each(function () {
        if ($(this).hasAttr("showonvalue")) {
            $(this).change(function () {
                onSelectHandler(this.id);
            });
            onSelectHandler(this.id);
        }
    });

    // Install select handler
    $('textarea').each(function () {
        if ($(this).hasAttr("maxchars")) {
            $(this).keydown(function () {
                onTextAreaKeyDownHandler(this.id);
            });
            $(this).focus(function () {
                onTextAreaFocusHandler(this.id);
            });
            $(this).blur(function () {
                onTextAreaBlurHandler(this.id);
            });
        }
    });

    // Activate JS version is lacking HTML5
    if (!HTML5_Placeholder)
        JSPlaceholder();
    if (!HTML5_Autofocus)
        JSAutofocus();

    // Add form validation to the form on submit
    $('form').submit(function () {
        return FormValidation($('form'));
    });

    // Define the FormError class
    $("<style type='text/css'> .FormError{ border: 2px solid red !important; } </style>").appendTo("head");
});

function InputMoneyOnly(e) {
    if ((e.shiftKey && e.keyCode == 45) || e.which != 8 && e.which != 0 && (e.which < 48 || e.which > 57) && e.which != 46 && e.which != 36) {
        return false;
    }
    // . = 46
    // , = 44
    // $ = 36
    var text = $(this).val();

    // Dollar sign first char only
    if (e.which == 36 && text.length != 0) {
        return false;
    }

    // Only one decimal point
    if (e.which == 46 && text.indexOf('.') != - 1) {
        return false;
    }

    // Only 2 numbers after decimal
    if (text.indexOf('.') != - 1 && (text.length - text.indexOf('.')) > 2) {
        return false;
    }

    return true;
}

function InputNumberOnly(e) {
    if ((e.shiftKey && e.keyCode == 45) || e.which != 8 && e.which != 0 && (e.which < 48 || e.which > 57) && e.which != 46) {
        return false;
    }

    // Only one decimal point
    var text = $(this).val();
    if (e.which == 46 && text.indexOf('.') != - 1) {
        return false;
    }
    return true;
}

// return true for 1234567890
function InputDigitsOnly(e) {
    if ((e.shiftKey && e.keyCode == 45) || e.which != 8 && e.which != 0 && (e.which < 48 || e.which > 57)) {
        return false;
    }
    return true;
}

// return true for 1234567890A-Za-z - _
function InputSlugOnly(e) {
    if ((e.shiftKey && e.keyCode == 45) || e.which != 8 && e.which != 0 && (e.which < 48 || e.which > 57)) {
        if (e.which == 45 || e.which == 95 || (e.which >= 65 && e.which <= 90) || (e.which >= 97 && e.which <= 122))
            return true;
        return false;
    }
    return true;
}

// return true for 1234567890 ()-
function InputTelOnly(e) {
    if ((e.shiftKey && e.keyCode == 45) || e.which != 8 && e.which != 0 && (e.which < 48 || e.which > 57)) {
        if (e.which == 32 || e.which == 40 || e.which == 41 || e.which == 45 || e.which == 43) {
            var text = $(this).val();
            // . = 46
            // , = 44
            // $ = 36
            // ( = 40, ) = 41, - = 45, <space> = 32, + = 43

            // Plus sign first char only
            if (e.which == 43 && text.length != 0) {
                return false;
            }

            // Only one (
            if (e.which == 40 && text.indexOf('(') != - 1) {
                return false;
            }

            // Only one ) after a (
            if (e.which == 41 && (text.indexOf(')') != - 1 || text.indexOf('(') == - 1)) {
                return false;
            }

            return true;
        }
        return false;
    }

    return true;
}

// Format a 7 or 10 digit number, else, leave it alone
function InputTelFormat() {
    var placeholder = $(this).attr('placeholder');
    var input = $(this).val();
    if (input != placeholder) {
        var formated = input.replace(/ /g, "").replace(/\(/g, "").replace(/\)/g, "").replace(/-/g, "");
        if (formated.length == 10)
            $(this).val("(" + formated.substring(0, 3) + ") " + formated.substring(3, 6) + "-" + formated.substring(6, 10));
        else if (formated.length == 7)
            $(this).val(formated.substring(0, 3) + "-" + formated.substring(3, 7));
    }
}

// Impliment HTML5 placeholder attribute in Javascript
function JSPlaceholder() {
    $elms = $('input, textarea');

    $elms.each(function () {
        var placeholderText = $(this).attr('placeholder');
        if (placeholderText && placeholderText != "") {
            var elmColor = $(this).attr('originalColor');
            if (!elmColor) {
                elmColor = $(this).css('color');
                $(this).attr('originalColor', elmColor);
            }

            if ($(this).val() == '') {
                $(this).val(placeholderText);
                $(this).css('color', '#CCCCCC');
            }

            $(this).click(function () {
                var placeholderText = $(this).attr('placeholder');
                if (placeholderText && $(this).val() == placeholderText) {
                    var elmColor = $(this).attr('originalColor');
                    $(this).val('');
                    $(this).css('color', elmColor);
                }
            });

            $(this).focus(function () {
                var placeholderText = $(this).attr('placeholder');
                if (placeholderText && $(this).val() == placeholderText) {
                    var elmColor = $(this).attr('originalColor');
                    $(this).val('');
                    $(this).css('color', elmColor);
                }
            });

            $(this).blur(function () {
                var placeholderText = $(this).attr('placeholder');
                if (placeholderText && $(this).val() == '') {
                    $(this).val(placeholderText);
                    $(this).css('color', '#CCCCCC');
                }
            });

            $(this).change(function () {
                var placeholderText = $(this).attr('placeholder');
                if (placeholderText && $(this).val() == '') {
                    $(this).val(placeholderText);
                    $(this).css('color', '#CCCCCC');
                }
            });

        }
    });
}


// Impliment HTML5 autofocus attribute
function JSAutofocus() {
    var $elms = $('input,textarea,select');
    $elms.each(function (idx, elm) {
        var attr = $(elm).attr('autofocus');
        var autofocus = (typeof attr !== 'undefined' && attr !== false);
        if (autofocus) {
            $(elm).focus();
        }
    });
}


function onSelectHandler(dropDown) {
    var value = 0;
    var divID = 1;
    var requiredInputID = 2;

    if ($('#' + dropDown).hasAttr('showonvalue')) {
        var attr = $('#' + dropDown).attr("showonvalue");
        var split = attr.split('|');
        for (var i = 0; i < split.length; i++) {
            var cmd = split[i].split(',');
            if ($('#' + dropDown + ' option:selected').text() == cmd[value]) {
                $('#' + cmd[divID]).css('display', 'block');
                if (cmd[requiredInputID] != '') {
                    $('#' + cmd[requiredInputID]).attr('required', 'true');
                }

                // Update the fields with the required span if needed
                var $elms = $('#' + cmd[divID]).find('input, textarea, select');
                $elms.each(function () {
                    if ($(this).prev('span').length == 0) {
                        if ($(this).hasAttr('required')) {
                            var reqVal = $(this).attr('reqmark');
                            if (reqVal && reqVal != '')
                                $('<span>').html(reqVal).addClass('FormRequired').insertAfter(this);
                            else
                                $('<span>').html('&lowast;').addClass('FormRequired').insertAfter(this);
                        } else {
                            if ($(this).width() > 0) {
                                $('<span>').html('&nbsp;').addClass('FormNotRequired').insertAfter(this);
                            }
                        }
                    }
                });
            } else
            {
                $('#' + cmd[divID]).css('display', 'none');
                $('#' + cmd[requiredInputID]).removeAttr('required');
                var id = $('#' + cmd[requiredInputID]).attr('id') + 'ErrMsg';
                $('#' + cmd[requiredInputID]).removeClass('FormError');
                $('#' + id).remove();
            }
        }
    }
}

function onTextAreaFocusHandler(elmID) {
    var id = $('#' + elmID).attr('id') + 'CharCount';
    var position = $('#' + elmID).offset();
    var max = $('#' + elmID).attr('maxchars');

    if ($('#' + id).length == 0) {
        var $counter = $('<div>').attr('id', id).css('position', 'absolute').appendTo('body');
        $counter.html($('#' + elmID).val().length + '/' + max);
        $counter.addClass('FormCharacterCount');
        $counter.css('top', position.top - $('#' + id).height() - 5 + 'px');
        $counter.css('left', position.left + $('#' + elmID).width() - $('#' + id).width() + 'px');
    }
}

function onTextAreaBlurHandler(elmID) {
    var id = $('#' + elmID).attr('id') + 'CharCount';
    $('#' + id).remove();
}

function onTextAreaKeyDownHandler(elmID) {
    var id = $('#' + elmID).attr('id') + 'CharCount';
    var position = $('#' + elmID).offset();
    var max = $('#' + elmID).attr('maxchars');
    var $counter = $('#' + id);

    if ($('#' + elmID).val().length > max)
        $('#' + elmID).val($('#' + elmID).val().substr(0, max));

    $counter.html($('#' + elmID).val().length + '/' + max);
    $counter.css('top', position.top - $('#' + id).height() - 5 + 'px');
    $counter.css('left', position.left + $('#' + elmID).width() - $('#' + id).width() + 'px');

}

// Validate the form, returns true for OK, false for Not OK.
function FormValidation(form) {
    var retVal = true;
    var firstErrorID = "";
    $elms = form.find('input, textarea, select');

    $elms.each(function () {
        // Remove the placeholder if needed
        var placeholderText = $(this).attr('placeholder');
        if (placeholderText && placeholderText != "") {
            if ($(this).val() == placeholderText)
                $(this).val('');
        }

        // Grab / Store the original background color
        var elmBG = $(this).attr('originalBG');
        if (!elmBG) {
            elmBG = $(this).css('border');
            $(this).attr('originalBG', elmBG);
        }

        // Grab the input type, value and required attributes
        var elmType = $(this).attr('xtype');
        var elmValue = "" + $(this).val();
        if (!elmType) {
            elmType = $(this).attr('type');
        }

        var elmRequired = false;
        var strReq = "" + $(this).hasAttr('required'); // This line is needed
        strReq = strReq.toUpperCase();
        if (strReq == "TRUE") {
            elmRequired = true;
        }

        // If the field is required OR the field is not blank, run the validation
        var pass = true;
        var id = $(this).attr('id');
        if (elmRequired == true || $(this).val() != "") {
            // Telephone numbers
            if (elmType == 'text') {
                var filter = new RegExp("^.+$");
            } else if (elmType == 'password') {
                var filter = new RegExp("^.+$");
            } else if (elmType == 'time') {
                var filter = new RegExp("^.+$");
            } else if (elmType == 'tel') {
                var filter = new RegExp("^[+][0-9]|(([0-9]( |-)?)?(\\(?[0-9]{3}\\)?|[0-9]{3})( |-)?([0-9]{3}( |-)?[0-9]{4}|[a-zA-Z0-9]{7})$|^(\\d{3})[- ]?(\\d{4}))$");
            } else if (elmType == 'url') {
                var filter = new RegExp("^(http|ftp|https):\\/\\/[\\w\\-_]+(\\.[\\w\\-_]+)+([\\w\\-\\.,@?^=%&amp;:/~\\+#]*[\\w\\-\\@?^=%&amp;/~\\+#])?$");
            } else if (elmType == 'email') {
                var filter = new RegExp("^[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?\\.)+(?:[a-zA-Z]{2}|[cC][oO][mM]|[oO][rR][gG]|[nN][eE][tT]|[eE][dD][uU]|[gG][oO][vV]|[mM][iI][lL]|[bB][iI][zZ]|[iI][nN][fF][oO]|[mM][oO][bB][iI]|[nN][aA][mM][eE]|[aA][eE][rR][oO]|[aA][sS][iI][aA]|[jJ][oO][bB][sS]|[mM][uU][sS][eE][uU][mM])$");
            } else if (elmType == 'number') {
                var filter = new RegExp("^-?\\d[\\d,]*(?:\\.\\d+)?$|^-?\\.\\d+$");
            } else if (elmType == 'digits') {
                var filter = new RegExp("^\\d\\d*$");
            } else if (elmType == 'date') {
                // mm/dd/yyyy
                var filter = new RegExp("^(0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])[- /.](19|20)[0-9][0-9]$");
            }
            if (filter) {
                if (!elmValue.match(filter)) {
                    pass = false;
                }
            } else {
                if (elmValue == "" && elmRequired) {
                    pass = false;
                }
            }
        }

        // check for matching values
        var matchingID = $(this).attr('match');
        if (matchingID && matchingID != '') {
            var $m = $('#' + matchingID)
            var matchVal = $m.val();
            if ($(this).val() != matchVal) {
                pass = false;
            }
        }

        if (!pass) {
            retVal = false;
            $(this).addClass('FormError');

            var position = $(this).offset();
            var msg = $(this).attr('errmsg');
            var id = $(this).attr('id') + 'ErrMsg';

            if (msg && msg != '') {
                if ($('#' + id).length == 0) {
                    var $err = $('<div>').attr('id', id).css('position', 'absolute').appendTo('body');
                    $err.addClass('FormErrorMessage FormErrorAuto');
                    $err.height($(this).innerHeight());
                    $err.html(msg);
                    $err.css('top', position.top + 'px');
                    $err.css('left', position.left + $(this).width() + 10 + 'px');
                } else {
                    if ($('#' + id).html() == "") {
                        $('#' + id).html(msg);
                    }
                    $('#' + id).show();
                }
            }
        } else {
            $(this).removeClass('FormError');
            var id = $(this).attr('id') + 'ErrMsg';
            if ($('#' + id).hasClass('FormErrorAuto')) {
                $('#' + id).remove();
            } else {
                $('#' + id).hide();
            }
        }
    });

    if (!retVal) {
        // Put the placehold back in place
        if (!HTML5_Placeholder) {
            $elms.each(function () {
                var placeholderText = $(this).attr('placeholder');
                if (placeholderText && placeholderText != "") {
                    if ($(this).val() == '') {
                        $(this).val(placeholderText);
                        $(this).css('color', '#CCCCCC');
                    }
                }
            });
        }
    }

    return retVal;
}

// Reset all validation classes and messages
function FormReset(form) {
    $elms = form.find('input, textarea, select');
    $elms.each(function () {
        $(this).removeClass('FormError');

        var id = $(this).attr('id') + 'ErrMsg';
        if ($('#' + id).hasClass('FormErrorAuto')) {
            $('#' + id).remove();
        } else {
            $('#' + id).hide();
        }

        // Grab / Store the original background color
        var elmBG = $(this).attr('originalBG');

        // Remove any values
        if (!elmBG)
            $(this).css('background-color', elmBG);

        // Setup the placeholder text
        if (!HTML5_Placeholder) {
            var placeholderText = $(this).attr('placeholder');
            if (placeholderText && placeholderText != "") {
                if ($(this).val() == '') {
                    $(this).val(placeholderText);
                    $(this).css('color', '#CCCCCC');
                }
            }
        }
    });
}

// Redirect known mobile browsers to a givin URL
function rediretMobileTo(URL) {
    (function (a, b) {
        if (/android|avantgo|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|symbian|treo|up\.(browser|link)|vodafone|wap|windows (ce|phone)|xda|xiino/i.test(a) || /1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|e\-|e\/|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(di|rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|xda(\-|2|g)|yas\-|your|zeto|zte\-/i.test(a.substr(0, 4)))
            window.location = b
    })(navigator.userAgent || navigator.vendor || window.opera, URL);
}


// Encode a string to be passed on as a URL variable
function urlencode(str) {
    return escape(str).replace(/\+/g, '%2B').replace(/%20/g, '+').replace(/\*/g, '%2A').replace(/\//g, '%2F').replace(/@/g, '%40');
}

// prototypes and stuff
Date.prototype.setISO8601 = function (string) {
    // Split into major parts
    var parts = string.split('T');
    var strDate = parts[0];
    parts = parts[1].split('Z');
    var strTime = parts[0];
    var strZone = parts[1];

    // split date apart
    parts = strDate.split('-');
    var year = parts[0];
    var month = parts[1];
    var day = parts[2];

    // set date
    this.setYear(year);
    this.setMonth(Number(month) - 1);
    this.setDate(day);

    // set time
    if (strTime) {
        parts = strTime.split(':');
        var hours = parts[0];
        var mins = parts[1];
        parts = parts[2].split('.');
        var secs = parts[0];
        var msecs = parts[1];

        this.setHours(hours);
        this.setMinutes(mins);
        this.setSeconds(secs);
        this.setMilliseconds(Number("0." + msecs) * 1000);
    }
}

// Add trim to Strings for IE
if (typeof String.prototype.trim !== 'function') {
    String.prototype.trim = function () {
        return this.replace(/^\s+|\s+$/g, '');
    }
}

$.fn.hasAttr = function (attr) {
    for (var i = 0; i < this[0].attributes.length; i++) {
        if (this[0].attributes[i].nodeName == attr) {
            return true 
        }
    }
    return false;
};

