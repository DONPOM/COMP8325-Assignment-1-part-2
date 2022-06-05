function getBidAsk(bid, ask, decimals) {

    if (bid === "N/A" || ask === "N/A") {
        return bid;
    }

    if (decimals !== null && decimals !== undefined) {
        try {
            bid = parseFloat(bid).toFixed(decimals);
        }
        catch (ex) {
        }
        try {
            ask = parseFloat(ask).toFixed(decimals);
        }
        catch (ex_) {
        }
    }

    var charsChanged = false;
    subBid = bid.substring(0, 1);
    subAsk = ask.substring(0, 1);
    finishAsk = "/";
    var i = 0;
    while (i < bid.length && i < ask.length) {
        subBid = bid.substring(i, i + 1);
        subAsk = ask.substring(i, i + 1);
        if (charsChanged || (subBid !== subAsk)) {
            charsChanged = true;
            finishAsk = finishAsk + subAsk;
        }
        i++;
    }
    if (finishAsk !== "/") {

        if (finishAsk.substring(1, finishAsk.length).length < 2 && ask.split('.').length > 1 && ask.split('.')[1].substring(1, finishAsk.length).length) {
            return bid + "/" + ask.substring(ask.length - 2, ask.length);
        }
        else {
            return bid + finishAsk;
        }
    } else {
        return bid;
    }
}

function getLast(value, bid, ask, decimals) {

    if (decimals !== null && decimals !== undefined) {
        if (value !== undefined) {
            try {
                value = parseFloat(value).toFixed(decimals);
            }
            catch (ex) {
            }
        }
    }

    var val = value;
    if (val === undefined) {
        val = getBidAsk(bid, ask, decimals);
    }
    return val;
}

function computeChange(last, close) {
    var index = last.indexOf('/');
    var newLast = last;
    if (index !== -1) {
        newLast = last.split('/')[0];
    }
    return parseFloat(newLast - close);//.toFixed(6);
}

function computePctChange(last, close) {
    var index = last.indexOf('/');
    var newLast = last;
    if (index !== -1) {
        newLast = last.split('/')[0];
    }
    
    if (close !== "" && close !== 0) {
        return parseFloat((newLast - close) * 100 / close);//.toFixed(2);
    }
    return 'NaN';
}

function computeYChange(last, yclose) {
    var index = last.indexOf('/');
    var newLast = last;
    if (index !== -1) {
        newLast = last.split('/')[0];
    }
    return parseFloat(newLast - yclose).toFixed(2);
}

function computePctYChange(last, yclose) {
    var index = last.indexOf('/');
    var newLast = last;
    if (index !== -1) {
        newLast = last.split('/')[0];
    }
    if (yclose !== "" && yclose !== 0) {
        return parseFloat((newLast - yclose) * 100 / yclose).toFixed(2);
    }
    return 'NaN';
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

    var cell = document.getElementById(cellid);
    if (cell !== null && val!= oldval) {

        var torender = '<div>' + val + '</div>';
        if (grid.renderers[fxy] !== undefined) {
            torender = grid.renderers[fxy].fn(grid, rec, cellid, val, oldval, fxy);
        }

        cell.innerHTML = torender;

        if (grid.renderers[fxy] !== undefined && grid.renderers[fxy].hightlightInterval !== undefined) {
            setTimeout(function () {
                Netdania.Util.resetHighlight(cellid);
            }, grid.renderers[fxy].hightlightInterval);

        }
    }
}

function receiveQuotesJsonData(data, grid) {
    //ResetHighlightOnGrid(grid);
    var recs = grid.records;
    grid.columns = grid.columns || grid.fields;

    for (var k = 0; k < data.length; k++) {
        if (data[k].f) {

            var r = getRecordById(grid.records, "recid-" + data[k].i);

            if (r === undefined) {
                continue;
            }
            r.hasLast = r.hasLast || false;

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

            var close1W = 'notdirty';
            var close3M = 'notdirty';
            var close12M = 'notdirty';

            for (var i = 0; data[k] && i < data[k].f.length; i++) {
                var field = data[k].f[i];
                if (field.f === 6) {
                    r.hasLast = true;
                    r.last = parseFloat(field.v);
                    last = "dirty";
                }
                if (field.f === 10) {r.oldBid = r.bid;  r.bid = field.v; bid = 'dirty'; }
                if (field.f === 11) { r.oldAsk = r.ask; r.ask = field.v; ask = 'dirty'; }
                if (field.f === 1) { r.close = field.v; close = 'dirty'; }
                if (field.f === 19) {
                    r.yclose = field.v; yclose = 'dirty'; 
                }

                if (field.f === 27) { r.close1W = field.v; close1W = 'dirty'; }
                if (field.f === 29) { r.close3M = field.v; close3M = 'dirty'; }
                if (field.f === 30) {
                    r.close12M = field.v; close12M = 'dirty'; 
                }

                if (field.f === 1003) {
                    if (r.realSymbol !== undefined) {
                        realSymbol = "dirty";
                    }
                    r.realSymbol = field.v;
                }
                if (field.f === 39) { r.isin = field.v; }

                if ("f" + field.f in grid.columns) {
                    tempOldValue = r.data["f" + field.f];
                    r.data["f" + field.f] = field.v;
                    if (field.f === 6) {
                        r.hasLast = true;
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

            if (!r.hasLast && last === "notdirty" && (bid === "dirty" || ask === "dirty")) {
                r.last = getLast(undefined, r.bid, r.ask, 5);
                last = "dirty";
                if ('f6' in grid.columns) {
                    tempOldValue = r.data.f6;
                    r.data.f6 = r.last;
                    refreshCell(grid, r, r.id + '-f6', r.last, tempOldValue, 'f6');
                }
            } else if (last === "dirty") {
                r.last = getLast(r.last, r.bid, r.ask, 5);
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

            if (last !== "notdirty" && yclose !== "notdirty") {
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

            if (last !== "notdirty" && close1W !== "notdirty") {
                if ('f33' in grid.columns) {
                    tempOldValue = r.data.f33;
                    r.data.f33 = computePctYChange(r.last, r.close1W);
                    refreshCell(grid, r, r.id + '-f33', r.data.f33, tempOldValue, 'f33');
                }
            }

            if (last !== "notdirty" && close3M !== "notdirty") {
                if ('f45' in grid.columns) {
                    tempOldValue = r.data.f45;
                    r.data.f45 = computePctYChange(r.last, r.close3M);
                    refreshCell(grid, r, r.id + '-f45', r.data.f45, tempOldValue, 'f45');
                }
            }

            if (last !== "notdirty" && close12M !== "notdirty") {
                if ('f37' in grid.columns) {
                    tempOldValue = r.data.f37;
                    r.data.f37 = computePctYChange(r.last, r.close12M);
                    refreshCell(grid, r, r.id + '-f37', r.data.f37, tempOldValue, 'f37');
                }
            }

        }
    }
}

function ResetHighlightOnGrid(grid) {
    for (var i = 0; i < grid.records.length; i++) {
        var groups = $('[id^=' + grid.records[i].id + ']');
        $.each(groups, function (key, group) {
            if ($(group).attr("id").indexOf('f25') == -1)
                Netdania.Util.resetHighlight($(group).attr("id"));
            //inputs = group.('[id^=pre]'); 
        });
  
        //Netdania.Util.resetHighlight(document.getElementById(grid.records[k].id));
    }
}


function replaceHtml(el, html) {
    if (el === undefined || el === null) { return; }
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
}

defaultRenderers = {
    f14: function (grid, rec, cellid, val, oldval, fxy) {
        val = parseFloat(val).toFixed(2);
        if (val === "NaN") {
            return '<div></div>';
        }
        return '<div>' + val + '</div>';
    },
    f6: function (grid, rec, cellid, val, oldval, fxy) {

        val = rec.decimals ? parseFloat(val).toFixed(rec.decimals) : val;

        if (val === "NaN") {
            return '<div>Wait...</div>';
        }

        var value = '<div>' + (val.indexOf(".") !== -1 ? val.replace(/0*$/, '') : val) + '</div>';
        if (val > oldval) {
            value = '<div class="val-rise">' + (val.indexOf(".") !== -1 ? val.replace(/0*$/, '') : val) + '</div>';
        } else if (val < oldval) {
            value = '<div class="val-fall">' + (val.indexOf(".") !== -1 ? val.replace(/0*$/, '') : val) + '</div>';
        }
        return value;
    },
    f15: function (grid, rec, cellid, val, oldval, fxy) {
        var upImagePath = "/JsComponents/jsgeneral/images/up.gif";
        var downImagePath = "/JsComponents/jsgeneral/images/down.gif";
        val = parseFloat(val).toFixed(2);
        val = val.replace("NaN", "");
        if (val === "") {
            return '<div></div>';
        }
        val = val.replace("-0.00", "0.00");
        var value = '<div>' + val + '%&nbsp; &nbsp;</div>';
        if (val > 0) {
            value = '<div>' + val + '% <img src="' + upImagePath + '"></div>';
        } else if (val < 0) {
            value = '<div>' + val + '% <img src="' + downImagePath + '"></div>';
        }
        return value;
    },
    f17: function (grid, rec, cellid, val, oldval, fxy) {
        if (val === "N/A" || val === undefined) {
            return "N/A";
        }
        var dateUnf = new Date(val * 1000);
        return '<div>' + Netdania.Util.dateFormat(dateUnf, "dd-HH:MM:ss") + '</div>';
    }
};



function getValue(data, f) {
    if (data.f) {
        for (var i = 0; data && i < data.f.length; i++) {
            var field = data.f[i];
            if (field.f === f) {
                return field.v;
            }
        }
    }
}



function getSymbolByISIN(isins, isin) {
    var symbol = '';
    for (var currentisin in isins) {
        if (isins.hasOwnProperty(currentisin)) {
            if (isins[currentisin].isin === isin) {
                symbol = isins[currentisin].symbol;
                break;
            }
        }
    }
    return symbol;
}

function getInstrBySymbol(instrs, symbol, field) {
    for (var ci in instrs) {
        if (instrs.hasOwnProperty(ci)) {
            if (instrs[ci][field] === symbol) {
                return instrs[ci];
            }
        }
    }
}

Netdania.removeLastDecimalPoint = function (val) {
    if (val.substr(val.length - 1, val.length - 1) === '.') {
        return val.substr(0, val.length - 1);
    }
    return val;
};
