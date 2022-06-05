$(function() {
    $("html").bind("ajaxStart", function() {
        $(this).addClass('busy');
    }).bind("ajaxStop", function() {
        $(this).removeClass('busy');
    });
});
function trim(sTrim) {
    if (sTrim == null) {
        return ""
    }
    return sTrim.replace(/^\s+/g, '').replace(/\s+$/g, '');
};
function pad(num, size) {
    var s = num + "";
    while (s.length < size)
        s = "0" + s;
    return s;
}
function isNumeric(n) {
    var n2 = n;
    n = parseFloat(n);
    return (n != 'NaN' && n2 == n);
};
function isAlpha(sTest) {
    var sValidChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    var sChar;
    var bResult = true;
    if (sTest.length == 0)
        return false;
    sTest = sTest.toUpperCase();
    for (i = 0; i < sTest.length && bResult == true; i++) {
        sChar = sTest.charAt(i);
        if (sValidChars.indexOf(sChar)==-1) {
            bResult = false;
        }
    }
    return bResult;
}
function validateIsNumeric(sTest) {
    if (isNumeric(sTest) == "") {
        return false;
    };
    return true;
}
function validateEmail(sTest) {
    if (sTest == "") {
        return false;
    }
    var reEmail = /^(([^<>()[\]\\.,;:\s@\"]+(\.[^<>()[\]\\.,;:\s@\"]+)*)|(\".+\"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    return reEmail.test(sTest);
}
function validateIsDate(sTest) {
    try {
        var dTest = new Date(sTest)
        if (isNumeric(dTest.getTime()) == "") {
            return false;
        }
    } catch (e) {
        return false;
    };
    return true;
}
function validateIsTime(sTest) {
    try {
        if (trim(sTest) == "") {
            return false;
        }
        var dTest = new Date("01/01/1970 " + sTest)
        if (isNumeric(dTest.getTime()) == "") {
            return false;
        }
    } catch (e) {
        return false;
    };
    return true;
}
function encryptPassword(sPassword) {
    return b64_md5(sPassword) + "== ";
};
jQuery.fn.numericOnly = function(bAllowNegative) {
    var bAllowNegativeVal = false;
    if (bAllowNegative == true)
        bAllowNegativeVal = true;
    return this.each(function() {
        $(this).keydown(function(oEvent) {
            var nKey = oEvent.which || oEvent.keyCode;
            var sVal = $(this).val();
            if (!oEvent.altKey&&!oEvent.ctrlKey&&!oEvent.shiftKey) {
                if (nKey >= 48 && nKey <= 57 || nKey >= 96 && nKey <= 105 || nKey == 8 || nKey == 9 || nKey == 13 || nKey == 27 || nKey == 35 || nKey == 36 || nKey == 37 || nKey == 39 || nKey == 190 || nKey == 188 || nKey == 110 || nKey == 46)
                    return true;
                if (bAllowNegativeVal == true) {
                    if (nKey == 109 && sVal.indexOf("-") < 0)
                        $(this).val("-" + sVal);
                    if ((nKey == 107 || nKey == 109) && sVal.indexOf("-") >= 0)
                        $(this).val(sVal.substr(1));
                }
            }
        });
    });
};
jQuery.fn.telephoneOnly = function() {
    return this.each(function() {
        $(this).keydown(function(oEvent) {
            return fnCheckKeyCode(oEvent, $(this), false, true);
        });
        $(this).blur(function(oEvent) {
            if (isNumeric($(this).val()) == false) {
                $(this).val("1");
            }
        });
    });
}
jQuery.fn.integerOnly = function(bAllowNegative) {
    var bAllowNegativeVal = true;
    if (bAllowNegative == false)
        bAllowNegativeVal = false;
    return this.each(function() {
        $(this).keydown(function(oEvent) {
            return fnCheckKeyCode(oEvent, $(this), bAllowNegativeVal, false);
        });
        $(this).blur(function(oEvent) {
            if (isNumeric($(this).val()) == false) {
                var sValidChars = "0123456789";
                var sChar;
                var sTest = $(this).val();
                var sResult = "";
                if (sTest.length == 0)
                    return false;
                for (i = 0; i < sTest.length; i++) {
                    sChar = sTest.charAt(i);
                    if (sValidChars.indexOf(sChar)!=-1) {
                        sResult = sResult + sChar;
                    }
                }
                if (sResult == "") {
                    sResult = "1";
                }
                $(this).val(sResult);
                return false;
            } else {
                return true;
            }
        });
    });
}
var fnCheckKeyCode = function(oEvent, oThis, bAllowNegative, bAllowTel) {
    var nKey = oEvent.which || oEvent.keyCode;
    var sVal = oThis.val();
    if (bAllowTel) {
        if ((oEvent.shiftKey && nKey == 61) || nKey == 107 || nKey == 48 || nKey == 57)
            return true;
    }
    if (!oEvent.altKey&&!oEvent.ctrlKey&&!oEvent.shiftKey) {
        if (nKey >= 48 && nKey <= 57 || nKey >= 96 && nKey <= 105 || nKey == 8 || nKey == 9 || nKey == 13 || nKey == 27 || nKey == 35 || nKey == 36 || nKey == 37 || nKey == 39 || nKey == 38 || nKey == 40 || nKey == 46)
            return true;
        if (bAllowNegative == true) {
            if (nKey == 109 && sVal.indexOf("-") < 0)
                oThis.val("-" + sVal);
            if ((nKey == 107 || nKey == 109) && sVal.indexOf("-") >= 0)
                oThis.val(sVal.substr(1));
        }
    }
    return false;
};
jQuery.fn.currencyOnly = function(bAllowNegative) {
    var bAllowNegativeVal = true;
    if (bAllowNegative == false)
        bAllowNegativeVal = false;
    return this.each(function() {
        $(this).keydown(function(oEvent) {
            var nKey = oEvent.which || oEvent.keyCode;
            var sVal = $(this).val();
            var iPos = sVal.lastIndexOf(".");
            var iMaxPlaces = 2;
            var iCurPlaces = 0;
            if (iPos >= 0)
                iCurPlaces = sVal.length - iPos;
            if (!oEvent.shiftKey && (nKey == 46 || nKey == 45))
                return true;
            if (nKey == 8 || nKey == 9 || nKey == 13 || nKey == 27 || nKey == 35 || nKey == 36 || nKey == 37 || nKey == 39 || nKey >= 112 && nKey <= 123)
                return true;
            if (iCurPlaces <= iMaxPlaces && (nKey >= 48 && nKey <= 57 || nKey >= 96 && nKey <= 105))
                return true;
            if (iPos < 0 && (nKey == 190 || nKey == 46 || nKey == 110))
                return true;
            if (bAllowNegativeVal == true) {
                if (nKey == 109 && sVal.indexOf("-") < 0)
                    $(this).val("-" + sVal);
                if ((nKey == 107 || nKey == 109) && sVal.indexOf("-") >= 0)
                    $(this).val(sVal.substr(1));
            }
            return false;
        });
    });
};
jQuery.fn.alphaOnly = function() {
    $(this).keydown(function(oEvent) {
        var nKey = oEvent.which || oEvent.keyCode;
        var sChar = String.fromCharCode(nKey).toUpperCase();
        if (nKey == 8 || nKey == 9 || nKey == 13 || nKey == 27 || nKey == 35 || nKey == 36 || nKey == 37 || nKey == 39 || nKey == 46 || nKey == 45 || nKey >= 112 && nKey <= 123)
            return true;
        if (nKey >= 48 && nKey <= 57 || nKey >= 96 && nKey <= 105)
            return false;
        var sValidChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        if (sValidChars.indexOf(sChar) >= 0) {
            return true;
        }
        return false;
    });
};
var typeaheadSubstringMatcher = function(strs) {
    return function findMatches(q, cb) {
        var matches, substringRegex;
        matches = [];
        substrRegex = new RegExp(q, 'i');
        $.each(strs, function(i, str) {
            if (substrRegex.test(str)) {
                matches.push({
                    value: str
                });
            }
        });
        cb(matches);
    };
};
