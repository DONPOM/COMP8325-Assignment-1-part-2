
if (typeof(console) == 'undefined')
    console = {
        log: function(obj) {}
    };
var dateFormat = function() {
    var token = /d{1,4}|m{1,4}|yy(?:yy)?|([HhMsTt])\1?|[LloZ]|"[^"]*"|'[^']*'/g, timezone = /\b(?:[PMCEA][SDP]T|(?:Pacific|Mountain|Central|Eastern|Atlantic) (?:Standard|Daylight|Prevailing) Time|(?:GMT|UTC)(?:[-+]\d{4})?)\b/g, timezoneClip = /[^-+\dA-Z]/g, pad = function(value, length) {
        value = String(value);
        length = parseInt(length) || 2;
        while (value.length < length)
            value = "0" + value;
        return value;
    };
    return function(date, mask) {
        if (arguments.length == 1 && (typeof date == "string" || date instanceof String)&&!/\d/.test(date)) {
            mask = date;
            date = undefined;
        }
        date = date ? new Date(date) : new Date();
        if (isNaN(date))
            throw "invalid date";
        var dF = dateFormat;
        mask = String(dF.masks[mask] || mask || dF.masks["default"]);
        var d = date.getDate(), D = date.getDay(), m = date.getMonth(), y = date.getFullYear(), H = date.getHours(), M = date.getMinutes(), s = date.getSeconds(), L = date.getMilliseconds(), o = date.getTimezoneOffset(), flags = {
            d: d,
            dd: pad(d),
            ddd: dF.i18n.dayNames[D],
            dddd: dF.i18n.dayNames[D + 7],
            m: m + 1,
            mm: pad(m + 1),
            mmm: dF.i18n.monthNames[m],
            mmmm: dF.i18n.monthNames[m + 12],
            yy: String(y).slice(2),
            yyyy: y,
            h: H%12 || 12,
            hh: pad(H%12 || 12),
            H: H,
            HH: pad(H),
            M: M,
            MM: pad(M),
            s: s,
            ss: pad(s),
            l: pad(L, 3),
            L: pad(L > 99 ? Math.round(L / 10) : L),
            t: H < 12 ? "a": "p",
            tt: H < 12 ? "am": "pm",
            T: H < 12 ? "A": "P",
            TT: H < 12 ? "AM": "PM",
            Z: (String(date).match(timezone) || [""]).pop().replace(timezoneClip, ""),
            o: (o > 0 ? "-" : "+") + pad(Math.floor(Math.abs(o) / 60) * 100 + Math.abs(o)%60, 4)
        };
        return mask.replace(token, function($0) {
            return ($0 in flags) ? flags[$0] : $0.slice(1, $0.length - 1);
        });
    };
}();
dateFormat.masks = {
    "default": "ddd mmm d yyyy HH:MM:ss",
    shortDate: "m/d/yy",
    mediumDate: "mmm d, yyyy",
    longDate: "mmmm d, yyyy",
    fullDate: "dddd, mmmm d, yyyy",
    shortTime: "h:MM TT",
    mediumTime: "h:MM:ss TT",
    longTime: "h:MM:ss TT Z",
    isoDate: "yyyy-mm-dd",
    isoTime: "HH:MM:ss",
    isoDateTime: "yyyy-mm-dd'T'HH:MM:ss",
    isoFullDateTime: "yyyy-mm-dd'T'HH:MM:ss.lo"
};
dateFormat.i18n = {
    dayNames: ["Sun", "Mon", "Tue", "Wed", "Thr", "Fri", "Sat", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    monthNames: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
};
Date.prototype.format = function(mask) {
    return dateFormat(this, mask);
}
var StringBuilder = Class.create({
    initialize: function() {
        this.array = new Array();
    },
    Append: function(str) {
        this.array.push(str);
    },
    AppendStrings: function(arr) {
        this.array.push(arr.join(""));
    },
    AppendFormat: function(str, obj) {
        this.array.push(str.interpolate(obj));
    },
    AppendLine: function() {
        this.array.push("\n");
    },
    Length: function() {
        return this.array.join("").length;
    },
    ToString: function() {
        return this.array.join("");
    }
});
var SSMath = function() {
    var usa = function() {
        this.UPrice = function(risk, towin) {
            return (towin >= risk) ? (towin / risk) * 100 : (risk / towin)*-100;
        }
        this.URisk = function(price, towin) {
            return (price > 0) ? (towin / price) * 100 : ( - price * towin) / 100;
        }
        this.UToWin = function(price, risk) {
            return (price > 0) ? (price * risk) / 100 : (risk/-price) * 100;
        }
    }
    this.USAOdds = new usa();
    var dec = function() {
        this.UPrice = function(risk, towin) {
            return (towin / risk) + 1;
        }
        this.URisk = function(price, towin) {
            return towin / (price - 1);
        }
        this.UToWin = function(price, risk) {
            return (price * risk) - risk;
        }
    }
    this.Decimal = new dec();
    this.ToCurrency = function(num) {
        num = num.toString().replace(/\$|\,/g, '');
        if (isNaN(num))
            num = "0";
        sign = (num == (num = Math.abs(num)));
        num = Math.floor(num * 100 + 0.50000000001);
        cents = num%100;
        num = Math.floor(num / 100).toString();
        if (cents < 10)
            cents = "0" + cents;
        for (var i = 0; i < Math.floor((num.length - (1 + i)) / 3); i++)
            num = num.substring(0, num.length - (4 * i + 3)) + ',' + num.substring(num.length - (4 * i + 3));
        return (((sign) ? '' : '-') + num + '.' + cents);
    }
}
var ModuleBox = {
    Toggle: function(obj, id) {
        var content = $(id);
        new Effect.toggle(content, 'blind', {
            duration: .3,
            afterFinish: function() {
                var visible = content.getStyle('display').toLowerCase() == 'block';
                obj.className = visible ? "moduleBox_arrowDown" : "moduleBox_arrowRight";
            }
        });
    }
}
var LiveHelpModule = {
    Toggle: function(obj, id) {
        $$('div.module_liveHelp').each(function(div) {
            div.hide();
        });
        $(id).show();
        $(obj).parentNode.select('li').each(function(li) {
            li.removeClassName('leagueTabSelected');
        });
        $(obj).addClassName('leagueTabSelected');
    }
}
var Forms = {
    CreateEnterActions: function() {
        Event.observe(window, 'load', function() {
            $(document.body).select('input[type="text"]', 'input[type="password"]').each(function(textbox) {
                if (textbox.getAttribute('buttonRel') != null) {
                    var buttonId = textbox.getAttribute('buttonRel');
                    Event.observe(textbox, 'keydown', function(event) {
                        if (event.which || event.keyCode) {
                            if ((event.which == 13) || (event.keyCode == 13)) {
                                $(buttonId).click();
                                return false;
                            }
                        } else
                            return true;
                    }.bind(this));
                }
            });
        });
    }
}
Forms.CreateEnterActions();
Date.CultureInfo = {
    name: "en-US",
    englishName: "English (United States)",
    nativeName: "English (United States)",
    dayNames: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    abbreviatedDayNames: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    shortestDayNames: ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"],
    firstLetterDayNames: ["S", "M", "T", "W", "T", "F", "S"],
    monthNames: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    abbreviatedMonthNames: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    amDesignator: "AM",
    pmDesignator: "PM",
    firstDayOfWeek: 0,
    twoDigitYearMax: 2029,
    dateElementOrder: "mdy",
    formatPatterns: {
        shortDate: "M/d/yyyy",
        longDate: "dddd, MMMM dd, yyyy",
        shortTime: "h:mm tt",
        longTime: "h:mm:ss tt",
        fullDateTime: "dddd, MMMM dd, yyyy h:mm:ss tt",
        sortableDateTime: "yyyy-MM-ddTHH:mm:ss",
        universalSortableDateTime: "yyyy-MM-dd HH:mm:ssZ",
        rfc1123: "ddd, dd MMM yyyy HH:mm:ss GMT",
        monthDay: "MMMM dd",
        yearMonth: "MMMM, yyyy"
    },
    regexPatterns: {
        jan: /^jan(uary)?/i,
        feb: /^feb(ruary)?/i,
        mar: /^mar(ch)?/i,
        apr: /^apr(il)?/i,
        may: /^may/i,
        jun: /^jun(e)?/i,
        jul: /^jul(y)?/i,
        aug: /^aug(ust)?/i,
        sep: /^sep(t(ember)?)?/i,
        oct: /^oct(ober)?/i,
        nov: /^nov(ember)?/i,
        dec: /^dec(ember)?/i,
        sun: /^su(n(day)?)?/i,
        mon: /^mo(n(day)?)?/i,
        tue: /^tu(e(s(day)?)?)?/i,
        wed: /^we(d(nesday)?)?/i,
        thu: /^th(u(r(s(day)?)?)?)?/i,
        fri: /^fr(i(day)?)?/i,
        sat: /^sa(t(urday)?)?/i,
        future: /^next/i,
        past: /^last|past|prev(ious)?/i,
        add: /^(\+|after|from)/i,
        subtract: /^(\-|before|ago)/i,
        yesterday: /^yesterday/i,
        today: /^t(oday)?/i,
        tomorrow: /^tomorrow/i,
        now: /^n(ow)?/i,
        millisecond: /^ms|milli(second)?s?/i,
        second: /^sec(ond)?s?/i,
        minute: /^min(ute)?s?/i,
        hour: /^h(ou)?rs?/i,
        week: /^w(ee)?k/i,
        month: /^m(o(nth)?s?)?/i,
        day: /^d(ays?)?/i,
        year: /^y((ea)?rs?)?/i,
        shortMeridian: /^(a|p)/i,
        longMeridian: /^(a\.?m?\.?|p\.?m?\.?)/i,
        timezone: /^((e(s|d)t|c(s|d)t|m(s|d)t|p(s|d)t)|((gmt)?\s*(\+|\-)\s*\d\d\d\d?)|gmt)/i,
        ordinalSuffix: /^\s*(st|nd|rd|th)/i,
        timeContext: /^\s*(\:|a|p)/i
    },
    abbreviatedTimeZoneStandard: {
        GMT: "-000",
        EST: "-0400",
        CST: "-0500",
        MST: "-0600",
        PST: "-0700"
    },
    abbreviatedTimeZoneDST: {
        GMT: "-000",
        EDT: "-0500",
        CDT: "-0600",
        MDT: "-0700",
        PDT: "-0800"
    }
};
Date.getMonthNumberFromName = function(name) {
    var n = Date.CultureInfo.monthNames, m = Date.CultureInfo.abbreviatedMonthNames, s = name.toLowerCase();
    for (var i = 0; i < n.length; i++) {
        if (n[i].toLowerCase() == s || m[i].toLowerCase() == s) {
            return i;
        }
    }
    return - 1;
};
Date.getDayNumberFromName = function(name) {
    var n = Date.CultureInfo.dayNames, m = Date.CultureInfo.abbreviatedDayNames, o = Date.CultureInfo.shortestDayNames, s = name.toLowerCase();
    for (var i = 0; i < n.length; i++) {
        if (n[i].toLowerCase() == s || m[i].toLowerCase() == s) {
            return i;
        }
    }
    return - 1;
};
Date.isLeapYear = function(year) {
    return (((year%4 === 0) && (year%100 !== 0)) || (year%400 === 0));
};
Date.getDaysInMonth = function(year, month) {
    return [31, (Date.isLeapYear(year) ? 29 : 28), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month];
};
Date.getTimezoneOffset = function(s, dst) {
    return (dst || false) ? Date.CultureInfo.abbreviatedTimeZoneDST[s.toUpperCase()] : Date.CultureInfo.abbreviatedTimeZoneStandard[s.toUpperCase()];
};
Date.getTimezoneAbbreviation = function(offset, dst) {
    var n = (dst || false) ? Date.CultureInfo.abbreviatedTimeZoneDST: Date.CultureInfo.abbreviatedTimeZoneStandard, p;
    for (p in n) {
        if (n[p] === offset) {
            return p;
        }
    }
    return null;
};
Date.prototype.clone = function() {
    return new Date(this.getTime());
};
Date.prototype.compareTo = function(date) {
    if (isNaN(this)) {
        throw new Error(this);
    }
    if (date instanceof Date&&!isNaN(date)) {
        return (this > date) ? 1 : (this < date)?-1 : 0;
    } else {
        throw new TypeError(date);
    }
};
Date.prototype.equals = function(date) {
    return (this.compareTo(date) === 0);
};
Date.prototype.between = function(start, end) {
    var t = this.getTime();
    return t >= start.getTime() && t <= end.getTime();
};
Date.prototype.addMilliseconds = function(value) {
    this.setMilliseconds(this.getMilliseconds() + value);
    return this;
};
Date.prototype.addSeconds = function(value) {
    return this.addMilliseconds(value * 1000);
};
Date.prototype.addMinutes = function(value) {
    return this.addMilliseconds(value * 60000);
};
Date.prototype.addHours = function(value) {
    return this.addMilliseconds(value * 3600000);
};
Date.prototype.addDays = function(value) {
    return this.addMilliseconds(value * 86400000);
};
Date.prototype.addWeeks = function(value) {
    return this.addMilliseconds(value * 604800000);
};
Date.prototype.addMonths = function(value) {
    var n = this.getDate();
    this.setDate(1);
    this.setMonth(this.getMonth() + value);
    this.setDate(Math.min(n, this.getDaysInMonth()));
    return this;
};
Date.prototype.addYears = function(value) {
    return this.addMonths(value * 12);
};
Date.prototype.add = function(config) {
    if (typeof config == "number") {
        this._orient = config;
        return this;
    }
    var x = config;
    if (x.millisecond || x.milliseconds) {
        this.addMilliseconds(x.millisecond || x.milliseconds);
    }
    if (x.second || x.seconds) {
        this.addSeconds(x.second || x.seconds);
    }
    if (x.minute || x.minutes) {
        this.addMinutes(x.minute || x.minutes);
    }
    if (x.hour || x.hours) {
        this.addHours(x.hour || x.hours);
    }
    if (x.month || x.months) {
        this.addMonths(x.month || x.months);
    }
    if (x.year || x.years) {
        this.addYears(x.year || x.years);
    }
    if (x.day || x.days) {
        this.addDays(x.day || x.days);
    }
    return this;
};
Date._validate = function(value, min, max, name) {
    if (typeof value != "number") {
        throw new TypeError(value + " is not a Number.");
    } else if (value < min || value > max) {
        throw new RangeError(value + " is not a valid value for " + name + ".");
    }
    return true;
};
Date.validateMillisecond = function(n) {
    return Date._validate(n, 0, 999, "milliseconds");
};
Date.validateSecond = function(n) {
    return Date._validate(n, 0, 59, "seconds");
};
Date.validateMinute = function(n) {
    return Date._validate(n, 0, 59, "minutes");
};
Date.validateHour = function(n) {
    return Date._validate(n, 0, 23, "hours");
};
Date.validateDay = function(n, year, month) {
    return Date._validate(n, 1, Date.getDaysInMonth(year, month), "days");
};
Date.validateMonth = function(n) {
    return Date._validate(n, 0, 11, "months");
};
Date.validateYear = function(n) {
    return Date._validate(n, 1, 9999, "seconds");
};
Date.prototype.set = function(config) {
    var x = config;
    if (!x.millisecond && x.millisecond !== 0) {
        x.millisecond =- 1;
    }
    if (!x.second && x.second !== 0) {
        x.second =- 1;
    }
    if (!x.minute && x.minute !== 0) {
        x.minute =- 1;
    }
    if (!x.hour && x.hour !== 0) {
        x.hour =- 1;
    }
    if (!x.day && x.day !== 0) {
        x.day =- 1;
    }
    if (!x.month && x.month !== 0) {
        x.month =- 1;
    }
    if (!x.year && x.year !== 0) {
        x.year =- 1;
    }
    if (x.millisecond!=-1 && Date.validateMillisecond(x.millisecond)) {
        this.addMilliseconds(x.millisecond - this.getMilliseconds());
    }
    if (x.second!=-1 && Date.validateSecond(x.second)) {
        this.addSeconds(x.second - this.getSeconds());
    }
    if (x.minute!=-1 && Date.validateMinute(x.minute)) {
        this.addMinutes(x.minute - this.getMinutes());
    }
    if (x.hour!=-1 && Date.validateHour(x.hour)) {
        this.addHours(x.hour - this.getHours());
    }
    if (x.month!==-1 && Date.validateMonth(x.month)) {
        this.addMonths(x.month - this.getMonth());
    }
    if (x.year!=-1 && Date.validateYear(x.year)) {
        this.addYears(x.year - this.getFullYear());
    }
    if (x.day!=-1 && Date.validateDay(x.day, this.getFullYear(), this.getMonth())) {
        this.addDays(x.day - this.getDate());
    }
    if (x.timezone) {
        this.setTimezone(x.timezone);
    }
    if (x.timezoneOffset) {
        this.setTimezoneOffset(x.timezoneOffset);
    }
    return this;
};
Date.prototype.clearTime = function() {
    this.setHours(0);
    this.setMinutes(0);
    this.setSeconds(0);
    this.setMilliseconds(0);
    return this;
};
Date.prototype.isLeapYear = function() {
    var y = this.getFullYear();
    return (((y%4 === 0) && (y%100 !== 0)) || (y%400 === 0));
};
Date.prototype.isWeekday = function() {
    return !(this.is().sat() || this.is().sun());
};
Date.prototype.getDaysInMonth = function() {
    return Date.getDaysInMonth(this.getFullYear(), this.getMonth());
};
Date.prototype.moveToFirstDayOfMonth = function() {
    return this.set({
        day: 1
    });
};
Date.prototype.moveToLastDayOfMonth = function() {
    return this.set({
        day: this.getDaysInMonth()
    });
};
Date.prototype.moveToDayOfWeek = function(day, orient) {
    var diff = (day - this.getDay() + 7 * (orient||+1))%7;
    return this.addDays((diff === 0) ? diff += 7 * (orient||+1) : diff);
};
Date.prototype.moveToMonth = function(month, orient) {
    var diff = (month - this.getMonth() + 12 * (orient||+1))%12;
    return this.addMonths((diff === 0) ? diff += 12 * (orient||+1) : diff);
};
Date.prototype.getDayOfYear = function() {
    return Math.floor((this - new Date(this.getFullYear(), 0, 1)) / 86400000);
};
Date.prototype.getWeekOfYear = function(firstDayOfWeek) {
    var y = this.getFullYear(), m = this.getMonth(), d = this.getDate();
    var dow = firstDayOfWeek || Date.CultureInfo.firstDayOfWeek;
    var offset = 7 + 1 - new Date(y, 0, 1).getDay();
    if (offset == 8) {
        offset = 1;
    }
    var daynum = ((Date.UTC(y, m, d, 0, 0, 0) - Date.UTC(y, 0, 1, 0, 0, 0)) / 86400000) + 1;
    var w = Math.floor((daynum - offset + 7) / 7);
    if (w === dow) {
        y--;
        var prevOffset = 7 + 1 - new Date(y, 0, 1).getDay();
        if (prevOffset == 2 || prevOffset == 8) {
            w = 53;
        } else {
            w = 52;
        }
    }
    return w;
};
Date.prototype.isDST = function() {
    return this.toString().match(/(E|C|M|P)(S|D)T/)[2] == "D";
};
Date.prototype.getTimezone = function() {
    return Date.getTimezoneAbbreviation(this.getUTCOffset, this.isDST());
};
Date.prototype.setTimezoneOffset = function(s) {
    var here = this.getTimezoneOffset(), there = Number(s)*-6 / 10;
    this.addMinutes(there - here);
    return this;
};
Date.prototype.setTimezone = function(s) {
    return this.setTimezoneOffset(Date.getTimezoneOffset(s));
};
Date.prototype.getUTCOffset = function() {
    var n = this.getTimezoneOffset()*-10 / 6, r;
    if (n < 0) {
        r = (n - 10000).toString();
        return r[0] + r.substr(2);
    } else {
        r = (n + 10000).toString();
        return "+" + r.substr(1);
    }
};
Date.prototype.getDayName = function(abbrev) {
    return abbrev ? Date.CultureInfo.abbreviatedDayNames[this.getDay()] : Date.CultureInfo.dayNames[this.getDay()];
};
Date.prototype.getMonthName = function(abbrev) {
    return abbrev ? Date.CultureInfo.abbreviatedMonthNames[this.getMonth()] : Date.CultureInfo.monthNames[this.getMonth()];
};
Date.prototype._toString = Date.prototype.toString;
Date.prototype.toString = function(format) {
    var self = this;
    var p = function p(s) {
        return (s.toString().length == 1) ? "0" + s : s;
    };
    return format ? format.replace(/dd?d?d?|MM?M?M?|yy?y?y?|hh?|HH?|mm?|ss?|tt?|zz?z?/g, function(format) {
        switch (format) {
        case"hh":
            return p(self.getHours() < 13 ? self.getHours() : (self.getHours() - 12));
        case"h":
            return self.getHours() < 13 ? self.getHours() : (self.getHours() - 12);
        case"HH":
            return p(self.getHours());
        case"H":
            return self.getHours();
        case"mm":
            return p(self.getMinutes());
        case"m":
            return self.getMinutes();
        case"ss":
            return p(self.getSeconds());
        case"s":
            return self.getSeconds();
        case"yyyy":
            return self.getFullYear();
        case"yy":
            return self.getFullYear().toString().substring(2, 4);
        case"dddd":
            return self.getDayName();
        case"ddd":
            return self.getDayName(true);
        case"dd":
            return p(self.getDate());
        case"d":
            return self.getDate().toString();
        case"MMMM":
            return self.getMonthName();
        case"MMM":
            return self.getMonthName(true);
        case"MM":
            return p((self.getMonth() + 1));
        case"M":
            return self.getMonth() + 1;
        case"t":
            return self.getHours() < 12 ? Date.CultureInfo.amDesignator.substring(0, 1) : Date.CultureInfo.pmDesignator.substring(0, 1);
        case"tt":
            return self.getHours() < 12 ? Date.CultureInfo.amDesignator : Date.CultureInfo.pmDesignator;
        case"zzz":
        case"zz":
        case"z":
            return "";
        }
    }) : this._toString();
};
Date.now = function() {
    return new Date();
};
Date.today = function() {
    return Date.now().clearTime();
};
Date.prototype._orient =+ 1;
Date.prototype.next = function() {
    this._orient =+ 1;
    return this;
};
Date.prototype.last = Date.prototype.prev = Date.prototype.previous = function() {
    this._orient =- 1;
    return this;
};
Date.prototype._is = false;
Date.prototype.is = function() {
    this._is = true;
    return this;
};
Number.prototype._dateElement = "day";
Number.prototype.fromNow = function() {
    var c = {};
    c[this._dateElement] = this;
    return Date.now().add(c);
};
Number.prototype.ago = function() {
    var c = {};
    c[this._dateElement] = this*-1;
    return Date.now().add(c);
};
(function() {
    var $D = Date.prototype, $N = Number.prototype;
    var dx = ("sunday monday tuesday wednesday thursday friday saturday").split(/\s/), mx = ("january february march april may june july august september october november december").split(/\s/), px = ("Millisecond Second Minute Hour Day Week Month Year").split(/\s/), de;
    var df = function(n) {
        return function() {
            if (this._is) {
                this._is = false;
                return this.getDay() == n;
            }
            return this.moveToDayOfWeek(n, this._orient);
        };
    };
    for (var i = 0; i < dx.length; i++) {
        $D[dx[i]] = $D[dx[i].substring(0, 3)] = df(i);
    }
    var mf = function(n) {
        return function() {
            if (this._is) {
                this._is = false;
                return this.getMonth() === n;
            }
            return this.moveToMonth(n, this._orient);
        };
    };
    for (var j = 0; j < mx.length; j++) {
        $D[mx[j]] = $D[mx[j].substring(0, 3)] = mf(j);
    }
    var ef = function(j) {
        return function() {
            if (j.substring(j.length - 1) != "s") {
                j += "s";
            }
            return this["add" + j](this._orient);
        };
    };
    var nf = function(n) {
        return function() {
            this._dateElement = n;
            return this;
        };
    };
    for (var k = 0; k < px.length; k++) {
        de = px[k].toLowerCase();
        $D[de] = $D[de + "s"] = ef(px[k]);
        $N[de] = $N[de + "s"] = nf(de);
    }
}());
Date.prototype.toJSONString = function() {
    return this.toString("yyyy-MM-ddThh:mm:ssZ");
};
Date.prototype.toShortDateString = function() {
    return this.toString(Date.CultureInfo.formatPatterns.shortDatePattern);
};
Date.prototype.toLongDateString = function() {
    return this.toString(Date.CultureInfo.formatPatterns.longDatePattern);
};
Date.prototype.toShortTimeString = function() {
    return this.toString(Date.CultureInfo.formatPatterns.shortTimePattern);
};
Date.prototype.toLongTimeString = function() {
    return this.toString(Date.CultureInfo.formatPatterns.longTimePattern);
};
Date.prototype.getOrdinal = function() {
    switch (this.getDate()) {
    case 1:
    case 21:
    case 31:
        return "st";
    case 2:
    case 22:
        return "nd";
    case 3:
    case 23:
        return "rd";
    default:
        return "th";
    }
};
(function() {
    Date.Parsing = {
        Exception: function(s) {
            this.message = "Parse error at '" + s.substring(0, 10) + " ...'";
        }
    };
    var $P = Date.Parsing;
    var _ = $P.Operators = {
        rtoken: function(r) {
            return function(s) {
                var mx = s.match(r);
                if (mx) {
                    return ([mx[0], s.substring(mx[0].length)]);
                } else {
                    throw new $P.Exception(s);
                }
            };
        },
        token: function(s) {
            return function(s) {
                return _.rtoken(new RegExp("^\s*" + s + "\s*"))(s);
            };
        },
        stoken: function(s) {
            return _.rtoken(new RegExp("^" + s));
        },
        until: function(p) {
            return function(s) {
                var qx = [], rx = null;
                while (s.length) {
                    try {
                        rx = p.call(this, s);
                    } catch (e) {
                        qx.push(rx[0]);
                        s = rx[1];
                        continue;
                    }
                    break;
                }
                return [qx, s];
            };
        },
        many: function(p) {
            return function(s) {
                var rx = [], r = null;
                while (s.length) {
                    try {
                        r = p.call(this, s);
                    } catch (e) {
                        return [rx, s];
                    }
                    rx.push(r[0]);
                    s = r[1];
                }
                return [rx, s];
            };
        },
        optional: function(p) {
            return function(s) {
                var r = null;
                try {
                    r = p.call(this, s);
                } catch (e) {
                    return [null, s];
                }
                return [r[0], r[1]];
            };
        },
        not: function(p) {
            return function(s) {
                try {
                    p.call(this, s);
                } catch (e) {
                    return [null, s];
                }
                throw new $P.Exception(s);
            };
        },
        ignore: function(p) {
            return p ? function(s) {
                var r = null;
                r = p.call(this, s);
                return [null, r[1]];
            } : null;
        },
        product: function() {
            var px = arguments[0], qx = Array.prototype.slice.call(arguments, 1), rx = [];
            for (var i = 0; i < px.length; i++) {
                rx.push(_.each(px[i], qx));
            }
            return rx;
        },
        cache: function(rule) {
            var cache = {}, r = null;
            return function(s) {
                try {
                    r = cache[s] = (cache[s] || rule.call(this, s));
                } catch (e) {
                    r = cache[s] = e;
                }
                if (r instanceof $P.Exception) {
                    throw r;
                } else {
                    return r;
                }
            };
        },
        any: function() {
            var px = arguments;
            return function(s) {
                var r = null;
                for (var i = 0; i < px.length; i++) {
                    if (px[i] == null) {
                        continue;
                    }
                    try {
                        r = (px[i].call(this, s));
                    } catch (e) {
                        r = null;
                    }
                    if (r) {
                        return r;
                    }
                }
                throw new $P.Exception(s);
            };
        },
        each: function() {
            var px = arguments;
            return function(s) {
                var rx = [], r = null;
                for (var i = 0; i < px.length; i++) {
                    if (px[i] == null) {
                        continue;
                    }
                    try {
                        r = (px[i].call(this, s));
                    } catch (e) {
                        throw new $P.Exception(s);
                    }
                    rx.push(r[0]);
                    s = r[1];
                }
                return [rx, s];
            };
        },
        all: function() {
            var px = arguments, _ = _;
            return _.each(_.optional(px));
        },
        sequence: function(px, d, c) {
            d = d || _.rtoken(/^\s*/);
            c = c || null;
            if (px.length == 1) {
                return px[0];
            }
            return function(s) {
                var r = null, q = null;
                var rx = [];
                for (var i = 0; i < px.length; i++) {
                    try {
                        r = px[i].call(this, s);
                    } catch (e) {
                        break;
                    }
                    rx.push(r[0]);
                    try {
                        q = d.call(this, r[1]);
                    } catch (ex) {
                        q = null;
                        break;
                    }
                    s = q[1];
                }
                if (!r) {
                    throw new $P.Exception(s);
                }
                if (q) {
                    throw new $P.Exception(q[1]);
                }
                if (c) {
                    try {
                        r = c.call(this, r[1]);
                    } catch (ey) {
                        throw new $P.Exception(r[1]);
                    }
                }
                return [rx, (r ? r[1] : s)];
            };
        },
        between: function(d1, p, d2) {
            d2 = d2 || d1;
            var _fn = _.each(_.ignore(d1), p, _.ignore(d2));
            return function(s) {
                var rx = _fn.call(this, s);
                return [[rx[0][0], r[0][2]], rx[1]];
            };
        },
        list: function(p, d, c) {
            d = d || _.rtoken(/^\s*/);
            c = c || null;
            return (p instanceof Array ? _.each(_.product(p.slice(0, - 1), _.ignore(d)), p.slice( - 1), _.ignore(c)) : _.each(_.many(_.each(p, _.ignore(d))), px, _.ignore(c)));
        },
        set: function(px, d, c) {
            d = d || _.rtoken(/^\s*/);
            c = c || null;
            return function(s) {
                var r = null, p = null, q = null, rx = null, best = [[], s], last = false;
                for (var i = 0; i < px.length; i++) {
                    q = null;
                    p = null;
                    r = null;
                    last = (px.length == 1);
                    try {
                        r = px[i].call(this, s);
                    } catch (e) {
                        continue;
                    }
                    rx = [[r[0]], r[1]];
                    if (r[1].length > 0&&!last) {
                        try {
                            q = d.call(this, r[1]);
                        } catch (ex) {
                            last = true;
                        }
                    } else {
                        last = true;
                    }
                    if (!last && q[1].length === 0) {
                        last = true;
                    }
                    if (!last) {
                        var qx = [];
                        for (var j = 0; j < px.length; j++) {
                            if (i != j) {
                                qx.push(px[j]);
                            }
                        }
                        p = _.set(qx, d).call(this, q[1]);
                        if (p[0].length > 0) {
                            rx[0] = rx[0].concat(p[0]);
                            rx[1] = p[1];
                        }
                    }
                    if (rx[1].length < best[1].length) {
                        best = rx;
                    }
                    if (best[1].length === 0) {
                        break;
                    }
                }
                if (best[0].length === 0) {
                    return best;
                }
                if (c) {
                    try {
                        q = c.call(this, best[1]);
                    } catch (ey) {
                        throw new $P.Exception(best[1]);
                    }
                    best[1] = q[1];
                }
                return best;
            };
        },
        forward: function(gr, fname) {
            return function(s) {
                return gr[fname].call(this, s);
            };
        },
        replace: function(rule, repl) {
            return function(s) {
                var r = rule.call(this, s);
                return [repl, r[1]];
            };
        },
        process: function(rule, fn) {
            return function(s) {
                var r = rule.call(this, s);
                return [fn.call(this, r[0]), r[1]];
            };
        },
        min: function(min, rule) {
            return function(s) {
                var rx = rule.call(this, s);
                if (rx[0].length < min) {
                    throw new $P.Exception(s);
                }
                return rx;
            };
        }
    };
    var _generator = function(op) {
        return function() {
            var args = null, rx = [];
            if (arguments.length > 1) {
                args = Array.prototype.slice.call(arguments);
            } else if (arguments[0]instanceof Array) {
                args = arguments[0];
            }
            if (args) {
                for (var i = 0, px = args.shift(); i < px.length; i++) {
                    args.unshift(px[i]);
                    rx.push(op.apply(null, args));
                    args.shift();
                    return rx;
                }
            } else {
                return op.apply(null, arguments);
            }
        };
    };
    var gx = "optional not ignore cache".split(/\s/);
    for (var i = 0; i < gx.length; i++) {
        _[gx[i]] = _generator(_[gx[i]]);
    }
    var _vector = function(op) {
        return function() {
            if (arguments[0]instanceof Array) {
                return op.apply(null, arguments[0]);
            } else {
                return op.apply(null, arguments);
            }
        };
    };
    var vx = "each any all".split(/\s/);
    for (var j = 0; j < vx.length; j++) {
        _[vx[j]] = _vector(_[vx[j]]);
    }
}());
(function() {
    var flattenAndCompact = function(ax) {
        var rx = [];
        for (var i = 0; i < ax.length; i++) {
            if (ax[i]instanceof Array) {
                rx = rx.concat(flattenAndCompact(ax[i]));
            } else {
                if (ax[i]) {
                    rx.push(ax[i]);
                }
            }
        }
        return rx;
    };
    Date.Grammar = {};
    Date.Translator = {
        hour: function(s) {
            return function() {
                this.hour = Number(s);
            };
        },
        minute: function(s) {
            return function() {
                this.minute = Number(s);
            };
        },
        second: function(s) {
            return function() {
                this.second = Number(s);
            };
        },
        meridian: function(s) {
            return function() {
                this.meridian = s.slice(0, 1).toLowerCase();
            };
        },
        timezone: function(s) {
            return function() {
                var n = s.replace(/[^\d\+\-]/g, "");
                if (n.length) {
                    this.timezoneOffset = Number(n);
                } else {
                    this.timezone = s.toLowerCase();
                }
            };
        },
        day: function(x) {
            var s = x[0];
            return function() {
                this.day = Number(s.match(/\d+/)[0]);
            };
        },
        month: function(s) {
            return function() {
                this.month = ((s.length == 3) ? Date.getMonthNumberFromName(s) : (Number(s) - 1));
            };
        },
        year: function(s) {
            return function() {
                var n = Number(s);
                this.year = ((s.length > 2) ? n : (n + (((n + 2000) < Date.CultureInfo.twoDigitYearMax) ? 2000 : 1900)));
            };
        },
        rday: function(s) {
            return function() {
                switch (s) {
                case"yesterday":
                    this.days =- 1;
                    break;
                case"tomorrow":
                    this.days = 1;
                    break;
                case"today":
                    this.days = 0;
                    break;
                case"now":
                    this.days = 0;
                    this.now = true;
                    break;
                }
            };
        },
        finishExact: function(x) {
            x = (x instanceof Array) ? x : [x];
            var now = new Date();
            this.year = now.getFullYear();
            this.month = now.getMonth();
            this.day = 1;
            this.hour = 0;
            this.minute = 0;
            this.second = 0;
            for (var i = 0; i < x.length; i++) {
                if (x[i]) {
                    x[i].call(this);
                }
            }
            this.hour = (this.meridian == "p" && this.hour < 13) ? this.hour + 12 : this.hour;
            if (this.day > Date.getDaysInMonth(this.year, this.month)) {
                throw new RangeError(this.day + " is not a valid value for days.");
            }
            var r = new Date(this.year, this.month, this.day, this.hour, this.minute, this.second);
            if (this.timezone) {
                r.set({
                    timezone: this.timezone
                });
            } else if (this.timezoneOffset) {
                r.set({
                    timezoneOffset: this.timezoneOffset
                });
            }
            return r;
        },
        finish: function(x) {
            x = (x instanceof Array) ? flattenAndCompact(x) : [x];
            if (x.length === 0) {
                return null;
            }
            for (var i = 0; i < x.length; i++) {
                if (typeof x[i] == "function") {
                    x[i].call(this);
                }
            }
            if (this.now) {
                return new Date();
            }
            var today = Date.today();
            var method = null;
            var expression=!!(this.days != null || this.orient || this.operator);
            if (expression) {
                var gap, mod, orient;
                orient = ((this.orient == "past" || this.operator == "subtract")?-1 : 1);
                if (this.weekday) {
                    this.unit = "day";
                    gap = (Date.getDayNumberFromName(this.weekday) - today.getDay());
                    mod = 7;
                    this.days = gap ? ((gap + (orient * mod))%mod) : (orient * mod);
                }
                if (this.month) {
                    this.unit = "month";
                    gap = (this.month - today.getMonth());
                    mod = 12;
                    this.months = gap ? ((gap + (orient * mod))%mod) : (orient * mod);
                    this.month = null;
                }
                if (!this.unit) {
                    this.unit = "day";
                }
                if (this[this.unit + "s"] == null || this.operator != null) {
                    if (!this.value) {
                        this.value = 1;
                    }
                    if (this.unit == "week") {
                        this.unit = "day";
                        this.value = this.value * 7;
                    }
                    this[this.unit + "s"] = this.value * orient;
                }
                return today.add(this);
            } else {
                if (this.meridian && this.hour) {
                    this.hour = (this.hour < 13 && this.meridian == "p") ? this.hour + 12 : this.hour;
                }
                if (this.weekday&&!this.day) {
                    this.day = (today.addDays((Date.getDayNumberFromName(this.weekday) - today.getDay()))).getDate();
                }
                if (this.month&&!this.day) {
                    this.day = 1;
                }
                return today.set(this);
            }
        }
    };
    var _ = Date.Parsing.Operators, g = Date.Grammar, t = Date.Translator, _fn;
    g.datePartDelimiter = _.rtoken(/^([\s\-\.\,\/\x27]+)/);
    g.timePartDelimiter = _.stoken(":");
    g.whiteSpace = _.rtoken(/^\s*/);
    g.generalDelimiter = _.rtoken(/^(([\s\,]|at|on)+)/);
    var _C = {};
    g.ctoken = function(keys) {
        var fn = _C[keys];
        if (!fn) {
            var c = Date.CultureInfo.regexPatterns;
            var kx = keys.split(/\s+/), px = [];
            for (var i = 0; i < kx.length; i++) {
                px.push(_.replace(_.rtoken(c[kx[i]]), kx[i]));
            }
            fn = _C[keys] = _.any.apply(null, px);
        }
        return fn;
    };
    g.ctoken2 = function(key) {
        return _.rtoken(Date.CultureInfo.regexPatterns[key]);
    };
    g.h = _.cache(_.process(_.rtoken(/^(0[0-9]|1[0-2]|[1-9])/), t.hour));
    g.hh = _.cache(_.process(_.rtoken(/^(0[0-9]|1[0-2])/), t.hour));
    g.H = _.cache(_.process(_.rtoken(/^([0-1][0-9]|2[0-3]|[0-9])/), t.hour));
    g.HH = _.cache(_.process(_.rtoken(/^([0-1][0-9]|2[0-3])/), t.hour));
    g.m = _.cache(_.process(_.rtoken(/^([0-5][0-9]|[0-9])/), t.minute));
    g.mm = _.cache(_.process(_.rtoken(/^[0-5][0-9]/), t.minute));
    g.s = _.cache(_.process(_.rtoken(/^([0-5][0-9]|[0-9])/), t.second));
    g.ss = _.cache(_.process(_.rtoken(/^[0-5][0-9]/), t.second));
    g.hms = _.cache(_.sequence([g.H, g.mm, g.ss], g.timePartDelimiter));
    g.t = _.cache(_.process(g.ctoken2("shortMeridian"), t.meridian));
    g.tt = _.cache(_.process(g.ctoken2("longMeridian"), t.meridian));
    g.z = _.cache(_.process(_.rtoken(/^(\+|\-)?\s*\d\d\d\d?/), t.timezone));
    g.zz = _.cache(_.process(_.rtoken(/^(\+|\-)\s*\d\d\d\d/), t.timezone));
    g.zzz = _.cache(_.process(g.ctoken2("timezone"), t.timezone));
    g.timeSuffix = _.each(_.ignore(g.whiteSpace), _.set([g.tt, g.zzz]));
    g.time = _.each(_.optional(_.ignore(_.stoken("T"))), g.hms, g.timeSuffix);
    g.d = _.cache(_.process(_.each(_.rtoken(/^([0-2]\d|3[0-1]|\d)/), _.optional(g.ctoken2("ordinalSuffix"))), t.day));
    g.dd = _.cache(_.process(_.each(_.rtoken(/^([0-2]\d|3[0-1])/), _.optional(g.ctoken2("ordinalSuffix"))), t.day));
    g.ddd = g.dddd = _.cache(_.process(g.ctoken("sun mon tue wed thu fri sat"), function(s) {
        return function() {
            this.weekday = s;
        };
    }));
    g.M = _.cache(_.process(_.rtoken(/^(1[0-2]|0\d|\d)/), t.month));
    g.MM = _.cache(_.process(_.rtoken(/^(1[0-2]|0\d)/), t.month));
    g.MMM = g.MMMM = _.cache(_.process(g.ctoken("jan feb mar apr may jun jul aug sep oct nov dec"), t.month));
    g.y = _.cache(_.process(_.rtoken(/^(\d\d?)/), t.year));
    g.yy = _.cache(_.process(_.rtoken(/^(\d\d)/), t.year));
    g.yyy = _.cache(_.process(_.rtoken(/^(\d\d?\d?\d?)/), t.year));
    g.yyyy = _.cache(_.process(_.rtoken(/^(\d\d\d\d)/), t.year));
    _fn = function() {
        return _.each(_.any.apply(null, arguments), _.not(g.ctoken2("timeContext")));
    };
    g.day = _fn(g.d, g.dd);
    g.month = _fn(g.M, g.MMM);
    g.year = _fn(g.yyyy, g.yy);
    g.orientation = _.process(g.ctoken("past future"), function(s) {
        return function() {
            this.orient = s;
        };
    });
    g.operator = _.process(g.ctoken("add subtract"), function(s) {
        return function() {
            this.operator = s;
        };
    });
    g.rday = _.process(g.ctoken("yesterday tomorrow today now"), t.rday);
    g.unit = _.process(g.ctoken("minute hour day week month year"), function(s) {
        return function() {
            this.unit = s;
        };
    });
    g.value = _.process(_.rtoken(/^\d\d?(st|nd|rd|th)?/), function(s) {
        return function() {
            this.value = s.replace(/\D/g, "");
        };
    });
    g.expression = _.set([g.rday, g.operator, g.value, g.unit, g.orientation, g.ddd, g.MMM]);
    _fn = function() {
        return _.set(arguments, g.datePartDelimiter);
    };
    g.mdy = _fn(g.ddd, g.month, g.day, g.year);
    g.ymd = _fn(g.ddd, g.year, g.month, g.day);
    g.dmy = _fn(g.ddd, g.day, g.month, g.year);
    g.date = function(s) {
        return ((g[Date.CultureInfo.dateElementOrder] || g.mdy).call(this, s));
    };
    g.format = _.process(_.many(_.any(_.process(_.rtoken(/^(dd?d?d?|MM?M?M?|yy?y?y?|hh?|HH?|mm?|ss?|tt?|zz?z?)/), function(fmt) {
        if (g[fmt]) {
            return g[fmt];
        } else {
            throw Date.Parsing.Exception(fmt);
        }
    }), _.process(_.rtoken(/^[^dMyhHmstz]+/), function(s) {
        return _.ignore(_.stoken(s));
    }))), function(rules) {
        return _.process(_.each.apply(null, rules), t.finishExact);
    });
    var _F = {};
    var _get = function(f) {
        return _F[f] = (_F[f] || g.format(f)[0]);
    };
    g.formats = function(fx) {
        if (fx instanceof Array) {
            var rx = [];
            for (var i = 0; i < fx.length; i++) {
                rx.push(_get(fx[i]));
            }
            return _.any.apply(null, rx);
        } else {
            return _get(fx);
        }
    };
    g._formats = g.formats(["yyyy-MM-ddTHH:mm:ss", "ddd, MMM dd, yyyy H:mm:ss tt", "ddd MMM d yyyy HH:mm:ss zzz", "d"]);
    g._start = _.process(_.set([g.date, g.time, g.expression], g.generalDelimiter, g.whiteSpace), t.finish);
    g.start = function(s) {
        try {
            var r = g._formats.call({}, s);
            if (r[1].length === 0) {
                return r;
            }
        } catch (e) {}
        return g._start.call({}, s);
    };
}());
Date._parse = Date.parse;
Date.parse = function(s) {
    var r = null;
    if (!s) {
        return null;
    }
    try {
        r = Date.Grammar.start.call({}, s);
    } catch (e) {
        return null;
    }
    return ((r[1].length === 0) ? r[0] : null);
};
Date.getParseFunction = function(fx) {
    var fn = Date.Grammar.formats(fx);
    return function(s) {
        var r = null;
        try {
            r = fn.call({}, s);
        } catch (e) {
            return null;
        }
        return ((r[1].length === 0) ? r[0] : null);
    };
};
Date.parseExact = function(s, fx) {
    return Date.getParseFunction(fx)(s);
};
CalendarDB = (function() {
    function y(at) {
        at = at || {};
        this.args = at = al(at, {
            animation: !c,
            cont: null,
            bottomBar: true,
            date: true,
            fdow: aq("fdow"),
            min: null,
            max: null,
            reverseWheel: false,
            selection: [],
            selectionType: y.SEL_SINGLE,
            weekNumbers: false,
            align: "Bl/ / /T/r",
            inputField: null,
            trigger: null,
            dateFormat: "%Y-%m-%d",
            opacity: j ? 1: 3,
            titleFormat: "%b %Y",
            showTime: false,
            timePos: "right",
            time: true,
            minuteStep: 5,
            disabled: ad,
            dateInfo: ad,
            onChange: ad,
            onSelect: ad,
            onTimeChange: ad,
            onFocus: ad,
            onBlur: ad
        });
        this.handlers = {};
        var P = this, D = new Date();
        at.min = Y(at.min);
        at.max = Y(at.max);
        if (at.date === true) {
            at.date = D
        }
        if (at.time === true) {
            at.time = D.getHours() * 100 + Math.floor(D.getMinutes() / at.minuteStep) * at.minuteStep
        }
        this.date = Y(at.date);
        this.time = at.time;
        this.fdow = at.fdow;
        m("onChange onSelect onTimeChange onFocus onBlur".split(/\s+/), function(av) {
            var au = at[av];
            if (!(au instanceof Array)) {
                au = [au]
            }
            P.handlers[av] = au
        });
        this.selection = new y.Selection(at.selection, at.selectionType, R, this);
        var ar = K.call(this);
        if (at.cont) {
            W(at.cont).appendChild(ar)
        }
        if (at.trigger) {
            this.manageFields(at.trigger, at.inputField, at.dateFormat)
        }
    }
    var S = navigator.userAgent, s = /opera/i.test(S), ai = /Konqueror|Safari|KHTML/i.test(S), j = /msie/i.test(S)&&!s&&!(/mac_powerpc/i.test(S)), c = j && /msie 6/i.test(S), x = /gecko/i.test(S)&&!ai&&!s&&!j, l = y.prototype, q = y.I18N = {};
    y.SEL_NONE = 0;
    y.SEL_SINGLE = 1;
    y.SEL_MULTIPLE = 2;
    y.SEL_WEEK = 3;
    y.dateToInt = V;
    y.intToDate = B;
    y.printDate = ab;
    y.formatString = k;
    y.i18n = aq;
    y.LANG = function(P, D, ar) {
        q.__ = q[P] = {
            name: D,
            data: ar
        }
    };
    y.setup = function(D) {
        return new y(D)
    };
    l.moveTo = function(aG, aD) {
        aG = Y(aG);
        var aC = af(aG, this.date, true), au, az = this.args, aH = az.min && af(aG, az.min), aI = az.max && af(aG, az.max);
        if (!az.animation) {
            aD = false
        }
        ae(aH != null && aH <= 1, [this.els.navPrevMonth, this.els.navPrevYear], "DynarchCalendar-navDisabled");
        ae(aI != null && aI>=-1, [this.els.navNextMonth, this.els.navNextYear], "DynarchCalendar-navDisabled");
        if (aH<-1) {
            aG = az.min;
            au = 1;
            aC = 0
        }
        if (aI > 1) {
            aG = az.max;
            au = 2;
            aC = 0
        }
        this.date = aG;
        this.refresh(!!aD);
        this.callHooks("onChange", this, aG, aD);
        if (aD&&!(aC == 0 && aD == 2)) {
            if (this._bodyAnim) {
                this._bodyAnim.stop()
            }
            var aw = this.els.body, ax = G("div", "DynarchCalendar-animBody-" + U[aC], aw), aF = aw.firstChild, av = am(aF) || 0.7, ay = au ? Z.brakes: aC == 0 ? Z.shake: Z.accel_ab2, aE = aC * aC > 4, ar = aE ? aF.offsetTop: aF.offsetLeft, aB = ax.style, aA = aE ? aw.offsetHeight: aw.offsetWidth;
            if (aC < 0) {
                aA += ar
            } else {
                if (aC > 0) {
                    aA = ar - aA
                } else {
                    aA = Math.round(aA / 7);
                    if (au == 2) {
                        aA =- aA
                    }
                }
            }
            if (!au && aC != 0) {
                var P = ax.cloneNode(true), D = P.style, at = 2 * aA;
                P.appendChild(aF.cloneNode(true));
                D[aE ? "marginTop": "marginLeft"] = aA + "px";
                aw.appendChild(P)
            }
            aF.style.visibility = "hidden";
            ax.innerHTML = ac(this);
            this._bodyAnim = ap({
                onUpdate: v(function(aK, aM) {
                    var aL = ay(aK);
                    if (P) {
                        var aJ = aM(aL, aA, at) + "px"
                    }
                    if (au) {
                        aB[aE ? "marginTop": "marginLeft"] = aM(aL, aA, 0) + "px"
                    } else {
                        if (aE || aC == 0) {
                            aB.marginTop = aM(aC == 0 ? ay(aK * aK) : aL, 0, aA) + "px";
                            if (aC != 0) {
                                D.marginTop = aJ
                            }
                        }
                        if (!aE || aC == 0) {
                            aB.marginLeft = aM(aL, 0, aA) + "px";
                            if (aC != 0) {
                                D.marginLeft = aJ
                            }
                        }
                    }
                    if (this.args.opacity > 2 && P) {
                        am(P, 1 - aL);
                        am(ax, aL)
                    }
                }, this),
                onStop: v(function(aJ) {
                    aw.innerHTML = ac(this, aG);
                    this._bodyAnim = null
                }, this)
            })
        }
        this._lastHoverDate = null;
        return aH>=-1 && aI <= 1
    };
    l.isDisabled = function(D) {
        var P = this.args;
        return (P.min && af(D, P.min) < 0) || (P.max && af(D, P.max) > 0) || P.disabled(D)
    };
    l.toggleMenu = function() {
        u(this, !this._menuVisible)
    };
    l.refresh = function(D) {
        var P = this.els;
        if (!D) {
            P.body.innerHTML = ac(this)
        }
        P.title.innerHTML = F(this);
        P.yearInput.value = this.date.getFullYear()
    };
    l.redraw = function() {
        var D = this.els;
        this.refresh();
        D.dayNames.innerHTML = h(this);
        D.menu.innerHTML = ak(this);
        if (D.bottomBar) {
            D.bottomBar.innerHTML = H(this)
        }
        t(D.topCont, v(function(ar) {
            var P = r[ar.className];
            if (P) {
                D[P] = ar
            }
            if (ar.className == "DynarchCalendar-menu-year") {
                p(ar, this._focusEvents);
                D.yearInput = ar
            } else {
                if (j) {
                    ar.setAttribute("unselectable", "on")
                }
            }
        }, this));
        this.setTime(null, true)
    };
    l.setLanguage = function(D) {
        var P = y.setLanguage(D);
        if (P) {
            this.fdow = P.data.fdow;
            this.redraw()
        }
    };
    y.setLanguage = function(D) {
        var P = q[D];
        if (P) {
            q.__ = P
        }
        return P
    };
    l.focus = function() {
        try {
            this.els[this._menuVisible ? "yearInput": "focusLink"].focus()
        } catch (D) {}
        i.call(this)
    };
    l.blur = function() {
        this.els.focusLink.blur();
        this.els.yearInput.blur();
        z.call(this)
    };
    l.showAt = function(P, ax, av) {
        if (this._showAnim) {
            this._showAnim.stop()
        }
        av = av && this.args.animation;
        var aw = this.els.topCont, ar = this, D = this.els.body.firstChild, au = D.offsetHeight, at = aw.style;
        at.position = "absolute";
        at.left = P + 223 + "px";
        at.top = ax + 2 + "px";
        at.zIndex = 10000;
        at.display = "";
        if (av) {
            D.style.marginTop =- au + "px";
            this.args.opacity > 1 && am(aw, 0);
            this._showAnim = ap({
                onUpdate: function(ay, az) {
                    D.style.marginTop =- az(Z.accel_b(ay), au, 0) + "px";
                    ar.args.opacity > 1 && am(aw, ay)
                },
                onStop: function() {
                    ar.args.opacity > 1 && am(aw, "");
                    ar._showAnim = null
                }
            })
        }
    };
    l.hide = function() {
        var at = this.els.topCont, P = this, D = this.els.body.firstChild, ar = D.offsetHeight, au = J(at).y;
        if (this.args.animation) {
            if (this._showAnim) {
                this._showAnim.stop()
            }
            this._showAnim = ap({
                onUpdate: function(av, aw) {
                    P.args.opacity > 1 && am(at, 1 - av);
                    D.style.marginTop =- aw(Z.accel_b(av), 0, ar) + "px";
                    at.style.top = aw(Z.accel_ab(av), au, au - 10) + "px"
                },
                onStop: function() {
                    at.style.display = "none";
                    D.style.marginTop = "";
                    P.args.opacity > 1 && am(at, "");
                    P._showAnim = null
                }
            })
        } else {
            at.style.display = "none"
        }
        this.inputField = null
    };
    l.popup = function(D, at) {
        D = W(D);
        if (!at) {
            at = this.args.align
        }
        at = at.split(/\x2f/);
        var ar = J(D), aw = this.els.topCont, ay = aw.style, au, ax = X();
        ay.visibility = "hidden";
        ay.display = "";
        this.showAt(0, 0);
        document.body.appendChild(aw);
        au = {
            x: aw.offsetWidth,
            y: aw.offsetHeight
        };
        function P(az) {
            var aA = {
                x: av.x,
                y: av.y
            };
            if (!az) {
                return aA
            }
            if (/B/.test(az)) {
                aA.y += D.offsetHeight
            }
            if (/b/.test(az)) {
                aA.y += D.offsetHeight - au.y
            }
            if (/T/.test(az)) {
                aA.y -= au.y
            }
            if (/l/.test(az)) {
                aA.x -= au.x - D.offsetWidth
            }
            if (/L/.test(az)) {
                aA.x -= au.x
            }
            if (/R/.test(az)) {
                aA.x += D.offsetWidth
            }
            if (/c/i.test(az)) {
                aA.x += (D.offsetWidth - au.x) / 2
            }
            if (/m/i.test(az)) {
                aA.y += (D.offsetHeight - au.y) / 2
            }
            return aA
        }
        var av = ar;
        av = P(at[0]);
        if (av.y < ax.y) {
            av.y = ar.y;
            av = P(at[1])
        }
        if (av.x + au.x > ax.x + ax.w) {
            av.x = ar.x;
            av = P(at[2])
        }
        if (av.y + au.y > ax.y + ax.h) {
            av.y = ar.y;
            av = P(at[3])
        }
        if (av.x < ax.x) {
            av.x = ar.x;
            av = P(at[4])
        }
        this.showAt(av.x, av.y, true);
        ay.visibility = "";
        this.focus()
    };
    l.manageFields = function(ar, P, D) {
        P = W(P);
        p(W(ar), "click", v(function() {
            this.inputField = P;
            this.dateFormat = D;
            if (this.selection.type == y.SEL_SINGLE) {
                var aw, av, au, at;
                aw = /input|textarea/i.test(P.tagName) ? P.value : (P.innerText || P.textContent);
                if (aw) {
                    av = /(^|[^%])%[bBmo]/.exec(D);
                    au = /(^|[^%])%[de]/.exec(D);
                    if (av && au) {
                        at = av.index < au.index
                    }
                    aw = CalendarDB.parseDate(aw.replace(/\s/g, ""), at);
                    if (aw) {
                        this.moveTo(aw);
                        this.selection.set(aw, false, true);
                        this.setTime(100 * aw.getHours() + aw.getMinutes());
                    }
                }
            }
            this.popup(ar)
        }, this))
    };
    l.callHooks = function(ar) {
        var at = b(arguments, 1), D = this.handlers[ar], P = 0;
        for (; P < D.length; ++P) {
            D[P].apply(this, at)
        }
    };
    l.addEventListener = function(P, D) {
        this.handlers[P].push(D)
    };
    l.removeEventListener = function(at, ar) {
        var D = this.handlers[at], P = D.length;
        while (--P >= 0) {
            if (D[P] === ar) {
                D.splice(P, 1)
            }
        }
    };
    l.getTime = function() {
        return this.time
    };
    l.setTime = function(au, P) {
        if (this.args.showTime) {
            au = this.time = au != null ? au : this.time;
            var ar = this.getHours(), D = this.getMinutes(), at = ar < 12;
            if (this.args.showTime == 12) {
                if (ar == 0) {
                    ar = 12
                }
                if (ar > 12) {
                    ar -= 12
                }
                this.els.timeAM.innerHTML = aq(at ? "AM" : "PM")
            }
            if (ar < 10) {
                ar = "0" + ar
            }
            if (D < 10) {
                D = "0" + D
            }
            this.els.timeHour.innerHTML = ar;
            this.els.timeMinute.innerHTML = D;
            if (!P) {
                this.callHooks("onTimeChange", this, au)
            }
        }
    };
    l.getHours = function() {
        return Math.floor(this.time / 100)
    };
    l.getMinutes = function() {
        return this.time%100
    };
    l.setHours = function(D) {
        if (D < 0) {
            D += 24
        }
        this.setTime(100 * (D%24) + this.time%100)
    };
    l.setMinutes = function(D) {
        if (D < 0) {
            D += 60
        }
        this.setTime(100 * this.getHours() + (D%60))
    };
    l._getInputYear = function() {
        var D = parseInt(this.els.yearInput.value, 10);
        if (isNaN(D)) {
            D = this.date.getFullYear()
        }
        return D
    };
    l._showTooltip = function(D) {
        var P = "", at, ar = this.els.tooltip;
        if (D) {
            D = B(D);
            at = this.args.dateInfo(D);
            if (at && at.tooltip) {
                P = "<div class='DynarchCalendar-tooltipCont'>" + ab(D, at.tooltip) + "</div>"
            }
        }
        ar.innerHTML = P
    };
    var ah = " align='center' cellspacing='0' cellpadding='0'";
    function h(D) {
        var ar = ["<table", ah, "><tr>"], P = 0;
        if (D.args.weekNumbers) {
            ar.push("<td><div class='DynarchCalendar-weekNumber'>", aq("wk"), "</div></td>")
        }
        while (P < 7) {
            var at = (P+++D.fdow)%7;
            ar.push("<td><div", aq("weekend").indexOf(at) >= 0 ? " class='DynarchCalendar-weekend'>" : ">", aq("sdn")[at], "</div></td>")
        }
        ar.push("</tr></table>");
        return ar.join("")
    }
    function ac(aw, aG, aD) {
        aG = aG || aw.date;
        aD = aD || aw.fdow;
        aG = new Date(aG);
        var aI = aG.getMonth(), av = [], aA = 0, D = aw.args.weekNumbers;
        aG.setDate(1);
        var az = (aG.getDay() - aD)%7;
        if (az < 0) {
            az += 7
        }
        aG.setDate( - az);
        aG.setDate(aG.getDate() + 1);
        var aE = new Date(), at = aE.getDate(), P = aE.getMonth(), aJ = aE.getFullYear();
        av[aA++] = "<table class='DynarchCalendar-bodyTable'" + ah + ">";
        for (var aC = 0; aC < 6; ++aC) {
            av[aA++] = "<tr class='DynarchCalendar-week";
            if (aC == 0) {
                av[aA++] = " DynarchCalendar-first-row"
            }
            if (aC == 5) {
                av[aA++] = " DynarchCalendar-last-row"
            }
            av[aA++] = "'>";
            if (D) {
                av[aA++] = "<td class='DynarchCalendar-first-col'><div class='DynarchCalendar-weekNumber'>" + a(aG) + "</div></td>"
            }
            for (var aB = 0; aB < 7; ++aB) {
                var aF = aG.getDate(), ay = aG.getMonth(), au = aG.getFullYear(), ar = 10000 * au + 100 * (ay + 1) + aF, aH = aw.selection.isSelected(ar), ax = aw.isDisabled(aG);
                av[aA++] = "<td class='";
                if (aB == 0&&!D) {
                    av[aA++] = " DynarchCalendar-first-col"
                }
                if (aB == 0 && aC == 0) {
                    aw._firstDateVisible = ar
                }
                if (aB == 6) {
                    av[aA++] = " DynarchCalendar-last-col";
                    if (aC == 5) {
                        aw._lastDateVisible = ar
                    }
                }
                if (aH) {
                    av[aA++] = " DynarchCalendar-td-selected"
                }
                av[aA++] = "'><div dyc-type='date' unselectable='on' dyc-date='" + ar + "' ";
                if (ax) {
                    av[aA++] = "disabled='1' "
                }
                av[aA++] = "class='DynarchCalendar-day";
                if (aq("weekend").indexOf(aG.getDay()) >= 0) {
                    av[aA++] = " DynarchCalendar-weekend"
                }
                if (ay != aI) {
                    av[aA++] = " DynarchCalendar-day-othermonth"
                }
                if (aF == at && ay == P && au == aJ) {
                    av[aA++] = " DynarchCalendar-day-today"
                }
                if (ax) {
                    av[aA++] = " DynarchCalendar-day-disabled"
                }
                if (aH) {
                    av[aA++] = " DynarchCalendar-day-selected"
                }
                ax = aw.args.dateInfo(aG);
                if (ax && ax.klass) {
                    av[aA++] = " " + ax.klass
                }
                av[aA++] = "'>" + aF + "</div></td>";
                aG.setDate(aF + 1)
            }
            av[aA++] = "</tr>"
        }
        av[aA++] = "</table>";
        return av.join("")
    }
    function n(D) {
        var P = ["<table class='DynarchCalendar-topCont'", ah, "><tr><td><div class='DynarchCalendar'>", !j ? "<button class='DynarchCalendar-focusLink'></button>": "<a class='DynarchCalendar-focusLink' href='#'></a>", "<div class='DynarchCalendar-topBar'><div dyc-type='nav' dyc-btn='-Y' dyc-cls='hover-navBtn,pressed-navBtn' class='DynarchCalendar-navBtn DynarchCalendar-prevYear'><div></div></div><div dyc-type='nav' dyc-btn='+Y' dyc-cls='hover-navBtn,pressed-navBtn' class='DynarchCalendar-navBtn DynarchCalendar-nextYear'><div></div></div><div dyc-type='nav' dyc-btn='-M' dyc-cls='hover-navBtn,pressed-navBtn' class='DynarchCalendar-navBtn DynarchCalendar-prevMonth'><div></div></div><div dyc-type='nav' dyc-btn='+M' dyc-cls='hover-navBtn,pressed-navBtn' class='DynarchCalendar-navBtn DynarchCalendar-nextMonth'><div></div></div><table class='DynarchCalendar-titleCont'", ah, "><tr><td><div dyc-type='title' dyc-btn='menu' dyc-cls='hover-title,pressed-title' class='DynarchCalendar-title'>", F(D), "</div></td></tr></table><div class='DynarchCalendar-dayNames'>", h(D), "</div></div><div class='DynarchCalendar-body'></div>"];
        if (D.args.bottomBar || D.args.showTime) {
            P.push("<div class='DynarchCalendar-bottomBar'>", H(D), "</div>")
        }
        P.push("<div class='DynarchCalendar-menu' style='display: none'>", ak(D), "</div><div class='DynarchCalendar-tooltip'></div></div></td></tr></table>");
        return P.join("")
    }
    function F(D) {
        return "<div unselectable='on'>" + ab(D.date, D.args.titleFormat) + "</div>"
    }
    function ak(P) {
        var au = ["<table height='100%'", ah, "><tr><td><table style='margin-top: 1.5em'", ah, "><tr><td colspan='3'><input dyc-btn='year' class='DynarchCalendar-menu-year' size='6' value='", P.date.getFullYear(), "' /></td></tr><tr><td><div dyc-type='menubtn' dyc-cls='hover-navBtn,pressed-navBtn' dyc-btn='today'>", aq("goToday"), "</div></td></tr></table><p class='DynarchCalendar-menu-sep'>&nbsp;</p><table class='DynarchCalendar-menu-mtable'", ah, ">"], av = aq("smn"), at = 0, D = au.length, ar;
        while (at < 12) {
            au[D++] = "<tr>";
            for (ar = 4; --ar > 0;) {
                au[D++] = "<td><div dyc-type='menubtn' dyc-cls='hover-navBtn,pressed-navBtn' dyc-btn='m" + at + "' class='DynarchCalendar-menu-month'>" + av[at++] + "</div></td>"
            }
            au[D++] = "</tr>"
        }
        au[D++] = "</table></td></tr></table>";
        return au.join("")
    }
    function w(D, P) {
        P.push("<table class='DynarchCalendar-time'" + ah + "><tr><td rowspan='2'><div dyc-type='time-hour' dyc-cls='hover-time,pressed-time' class='DynarchCalendar-time-hour'></div></td><td dyc-type='time-hour+' dyc-cls='hover-time,pressed-time' class='DynarchCalendar-time-up'></td><td rowspan='2' class='DynarchCalendar-time-sep'></td><td rowspan='2'><div dyc-type='time-min' dyc-cls='hover-time,pressed-time' class='DynarchCalendar-time-minute'></div></td><td dyc-type='time-min+' dyc-cls='hover-time,pressed-time' class='DynarchCalendar-time-up'></td>");
        if (D.args.showTime == 12) {
            P.push("<td rowspan='2' class='DynarchCalendar-time-sep'></td><td rowspan='2'><div class='DynarchCalendar-time-am' dyc-type='time-am' dyc-cls='hover-time,pressed-time'></div></td>")
        }
        P.push("</tr><tr><td dyc-type='time-hour-' dyc-cls='hover-time,pressed-time' class='DynarchCalendar-time-down'></td><td dyc-type='time-min-' dyc-cls='hover-time,pressed-time' class='DynarchCalendar-time-down'></td></tr></table>")
    }
    function H(D) {
        var ar = [], P = D.args;
        ar.push("<table", ah, " style='width:100%'><tr>");
        function at() {
            if (P.showTime) {
                ar.push("<td>");
                w(D, ar);
                ar.push("</td>")
            }
        }
        if (P.timePos == "left") {
            at()
        }
        if (P.bottomBar) {
            ar.push("<td>");
            ar.push("<table", ah, "><tr><td><div dyc-btn='today' dyc-cls='hover-bottomBar-today,pressed-bottomBar-today' dyc-type='bottomBar-today' class='DynarchCalendar-bottomBar-today'>", aq("today"), "</div></td></tr></table>");
            ar.push("</td>")
        }
        if (P.timePos == "right") {
            at()
        }
        ar.push("</tr></table>");
        return ar.join("")
    }
    var r = {
        "DynarchCalendar-topCont": "topCont",
        "DynarchCalendar-focusLink": "focusLink",
        DynarchCalendar: "main",
        "DynarchCalendar-topBar": "topBar",
        "DynarchCalendar-title": "title",
        "DynarchCalendar-dayNames": "dayNames",
        "DynarchCalendar-body": "body",
        "DynarchCalendar-menu": "menu",
        "DynarchCalendar-menu-year": "yearInput",
        "DynarchCalendar-bottomBar": "bottomBar",
        "DynarchCalendar-tooltip": "tooltip",
        "DynarchCalendar-time-hour": "timeHour",
        "DynarchCalendar-time-minute": "timeMinute",
        "DynarchCalendar-time-am": "timeAM",
        "DynarchCalendar-navBtn DynarchCalendar-prevYear": "navPrevYear",
        "DynarchCalendar-navBtn DynarchCalendar-nextYear": "navNextYear",
        "DynarchCalendar-navBtn DynarchCalendar-prevMonth": "navPrevMonth",
        "DynarchCalendar-navBtn DynarchCalendar-nextMonth": "navNextMonth"
    };
    function K() {
        var ar = G("div"), P = this.els = {}, D = {
            mousedown: v(I, this, true),
            mouseup: v(I, this, false),
            mouseover: v(T, this, true),
            mouseout: v(T, this, false),
            keypress: v(L, this)
        };
        D[x ? "DOMMouseScroll": "mousewheel"] = v(E, this);
        if (j) {
            D.dblclick = D.mousedown;
            D.keydown = D.keypress
        }
        ar.innerHTML = n(this);
        t(ar.firstChild, function(au) {
            var at = r[au.className];
            if (at) {
                P[at] = au
            }
            if (j) {
                au.setAttribute("unselectable", "on")
            }
        });
        p(P.main, D);
        p([P.focusLink, P.yearInput], this._focusEvents = {
            focus: v(i, this),
            blur: v(e, this)
        });
        this.moveTo(this.date, false);
        this.setTime(null, true);
        return P.topCont
    }
    function i() {
        if (this._bluringTimeout) {
            clearTimeout(this._bluringTimeout)
        }
        this.focused = true;
        M(this.els.main, "DynarchCalendar-focused");
        this.callHooks("onFocus", this)
    }
    function z() {
        this.focused = false;
        aj(this.els.main, "DynarchCalendar-focused");
        if (this._menuVisible) {
            u(this, false)
        }
        if (!this.args.cont) {
            this.hide()
        }
        this.callHooks("onBlur", this)
    }
    function e() {
        this._bluringTimeout = setTimeout(v(z, this), 50)
    }
    function N(D) {
        switch (D) {
        case"time-hour+":
            this.setHours(this.getHours() + 1);
            break;
        case"time-hour-":
            this.setHours(this.getHours() - 1);
            break;
        case"time-min+":
            this.setMinutes(this.getMinutes() + this.args.minuteStep);
            break;
        case"time-min-":
            this.setMinutes(this.getMinutes() - this.args.minuteStep);
            break;
        default:
            return 
        }
    }
    var U = {
        "-3": "backYear",
        "-2": "back",
        "0": "now",
        "2": "fwd",
        "3": "fwdYear"
    };
    function aa(P, at, D) {
        if (this._bodyAnim) {
            this._bodyAnim.stop()
        }
        var ar;
        if (at != 0) {
            ar = new Date(P.date);
            ar.setDate(1);
            switch (at) {
            case"-Y":
            case - 2:
                ar.setFullYear(ar.getFullYear() - 1);
                break;
            case"+Y":
            case 2:
                ar.setFullYear(ar.getFullYear() + 1);
                break;
            case"-M":
            case - 1:
                ar.setMonth(ar.getMonth() - 1);
                break;
            case"+M":
            case 1:
                ar.setMonth(ar.getMonth() + 1);
                break
            }
        } else {
            ar = new Date()
        }
        return P.moveTo(ar, !D)
    }
    function u(ar, P) {
        ar._menuVisible = P;
        ae(P, ar.els.title, "DynarchCalendar-pressed-title");
        var at = ar.els.menu;
        if (c) {
            at.style.height = ar.els.main.offsetHeight + "px"
        }
        if (!ar.args.animation) {
            O(at, P);
            if (ar.focused) {
                ar.focus()
            }
        } else {
            if (ar._menuAnim) {
                ar._menuAnim.stop()
            }
            var D = ar.els.main.offsetHeight;
            if (c) {
                at.style.width = ar.els.topBar.offsetWidth + "px"
            }
            if (P) {
                at.firstChild.style.marginTop =- D + "px";
                ar.args.opacity > 0 && am(at, 0);
                O(at, true)
            }
            ar._menuAnim = ap({
                onUpdate: function(au, av) {
                    at.firstChild.style.marginTop = av(Z.accel_b(au), - D, 0, !P) + "px";
                    ar.args.opacity > 0 && am(at, av(Z.accel_b(au), 0, 0.85, !P))
                },
                onStop: function() {
                    ar.args.opacity > 0 && am(at, 0.85);
                    at.firstChild.style.marginTop = "";
                    ar._menuAnim = null;
                    if (!P) {
                        O(at, false);
                        if (ar.focused) {
                            ar.focus()
                        }
                    }
                }
            })
        }
    }
    function I(az, ay) {
        ay = ay || window.event;
        var au = o(ay);
        if (au&&!au.getAttribute("disabled")) {
            var D = au.getAttribute("dyc-btn"), ax = au.getAttribute("dyc-type"), av = au.getAttribute("dyc-date"), at = this.selection, ar, P = {
                mouseover: an,
                mousemove: an,
                mouseup: function(aC) {
                    var aB = au.getAttribute("dyc-cls");
                    if (aB) {
                        aj(au, ao(aB, 1))
                    }
                    clearTimeout(ar);
                    d(document, P, true);
                    P = null
                }
            };
            if (az) {
                setTimeout(v(this.focus, this), 1);
                var aA = au.getAttribute("dyc-cls");
                if (aA) {
                    M(au, ao(aA, 1))
                }
                if ("menu" == D) {
                    this.toggleMenu()
                } else {
                    if (au && /^[+-][MY]$/.test(D)) {
                        if (aa(this, D)) {
                            var aw = v(function() {
                                if (aa(this, D, true)) {
                                    ar = setTimeout(aw, 40)
                                } else {
                                    P.mouseup();
                                    aa(this, D)
                                }
                            }, this);
                            ar = setTimeout(aw, 350);
                            p(document, P, true)
                        } else {
                            P.mouseup()
                        }
                    } else {
                        if ("year" == D) {
                            this.els.yearInput.focus();
                            this.els.yearInput.select()
                        } else {
                            if (ax == "time-am") {
                                p(document, P, true)
                            } else {
                                if (/^time/.test(ax)) {
                                    var aw = v(function(aB) {
                                        N.call(this, aB);
                                        ar = setTimeout(aw, 100)
                                    }, this, ax);
                                    N.call(this, ax);
                                    ar = setTimeout(aw, 350);
                                    p(document, P, true)
                                } else {
                                    if (av && at.type) {
                                        if (at.type == y.SEL_MULTIPLE) {
                                            if (ay.shiftKey && this._selRangeStart) {
                                                at.selectRange(this._selRangeStart, av)
                                            } else {
                                                if (!ay.ctrlKey&&!at.isSelected(av)) {
                                                    at.clear(true)
                                                }
                                                at.set(av, true);
                                                this._selRangeStart = av
                                            }
                                        } else {
                                            at.set(av);
                                            this.moveTo(B(av), 2)
                                        }
                                        au = this._getDateDiv(av);
                                        T.call(this, true, {
                                            target: au
                                        })
                                    }
                                    p(document, P, true)
                                }
                            }
                        }
                    }
                }
                if (j && P && /dbl/i.test(ay.type)) {
                    P.mouseup()
                }
                if (/^(DynarchCalendar-(topBar|bottomBar|weekend|weekNumber|menu(-sep)?))?$/.test(au.className)&&!this.args.cont) {
                    P.mousemove = v(g, this);
                    this._mouseDiff = f(ay, J(this.els.topCont));
                    p(document, P, true)
                }
            } else {
                if ("today" == D) {
                    if (!this._menuVisible && at.type == y.SEL_SINGLE) {
                        at.set(new Date())
                    }
                    this.moveTo(new Date(), true);
                    u(this, false)
                } else {
                    if (/^m([0-9]+)/.test(D)) {
                        var av = new Date(this.date);
                        av.setDate(1);
                        av.setMonth(RegExp.$1);
                        av.setFullYear(this._getInputYear());
                        this.moveTo(av, true);
                        u(this, false)
                    } else {
                        if (ax == "time-am") {
                            this.setHours(this.getHours() + 12)
                        }
                    }
                }
            }
            if (!j) {
                an(ay)
            }
        }
    }
    function g(P) {
        P = P || window.event;
        var D = this.els.topCont.style, ar = f(P, this._mouseDiff);
        D.left = ar.x + "px";
        D.top = ar.y + "px"
    }
    function o(P) {
        var D = P.target || P.srcElement, ar = D;
        while (D && D.getAttribute&&!D.getAttribute("dyc-type")) {
            D = D.parentNode
        }
        return D.getAttribute && D || ar
    }
    function ao(D, P) {
        return "DynarchCalendar-" + D.split(/,/)[P]
    }
    function T(au, at) {
        at = at || window.event;
        var ar = o(at);
        if (ar) {
            var P = ar.getAttribute("dyc-type");
            if (P&&!ar.getAttribute("disabled")) {
                if (!au ||!this._bodyAnim || P != "date") {
                    var D = ar.getAttribute("dyc-cls");
                    D = D ? ao(D, 0) : "DynarchCalendar-hover-" + P;
                    if (P != "date" || this.selection.type) {
                        ae(au, ar, D)
                    }
                    if (P == "date") {
                        ae(au, ar.parentNode.parentNode, "DynarchCalendar-hover-week");
                        this._showTooltip(ar.getAttribute("dyc-date"))
                    }
                    if (/^time-hour/.test(P)) {
                        ae(au, this.els.timeHour, "DynarchCalendar-hover-time")
                    }
                    if (/^time-min/.test(P)) {
                        ae(au, this.els.timeMinute, "DynarchCalendar-hover-time")
                    }
                    aj(this._getDateDiv(this._lastHoverDate), "DynarchCalendar-hover-date");
                    this._lastHoverDate = null
                }
            }
        }
        if (!au) {
            this._showTooltip()
        }
    }
    function E(ar) {
        ar = ar || window.event;
        var P = o(ar);
        if (P) {
            var at = P.getAttribute("dyc-btn"), D = P.getAttribute("dyc-type"), au = ar.wheelDelta ? ar.wheelDelta / 120: - ar.detail / 3;
            au = au < 0?-1 : au > 0 ? 1 : 0;
            if (this.args.reverseWheel) {
                au =- au
            }
            if (/^(time-(hour|min))/.test(D)) {
                switch (RegExp.$1) {
                case"time-hour":
                    this.setHours(this.getHours() + au);
                    break;
                case"time-min":
                    this.setMinutes(this.getMinutes() + this.args.minuteStep * au);
                    break
                }
                an(ar)
            } else {
                if (/Y/i.test(at)) {
                    au*=2
                }
                aa(this, - au);
                an(ar)
            }
        }
    }
    function R() {
        this.refresh();
        var D = this.inputField, P = this.selection;
        if (D) {
            var ar = P.print(this.dateFormat);
            (/input|textarea/i.test(D.tagName)) ? D.value = ar : D.innerHTML = ar
        }
        this.callHooks("onSelect", this, P)
    }
    var ag = {
        37: - 1,
        38: - 2,
        39: 1,
        40: 2
    }, Q = {
        33: - 1,
        34: 1
    };
    function L(aB) {
        if (this._menuAnim) {
            return 
        }
        aB = aB || window.event;
        var ar = aB.target || aB.srcElement, aC = ar.getAttribute("dyc-btn"), aD = aB.keyCode, ay = aB.charCode || aD, D = ag[aD];
        if ("year" == aC && aD == 13) {
            var au = new Date(this.date);
            au.setDate(1);
            au.setFullYear(this._getInputYear());
            this.moveTo(au, true);
            u(this, false);
            return an(aB)
        }
        if (this._menuVisible) {
            if (aD == 27) {
                u(this, false);
                return an(aB)
            }
        } else {
            if (!aB.ctrlKey) {
                D = null
            }
            if (D == null&&!aB.ctrlKey) {
                D = Q[aD]
            }
            if (aD == 36) {
                D = 0
            }
            if (D != null) {
                aa(this, D);
                return an(aB)
            }
            ay = String.fromCharCode(ay).toLowerCase();
            var ax = this.els.yearInput, P = this.selection;
            if (ay == " ") {
                u(this, true);
                this.focus();
                ax.focus();
                ax.select();
                return an(aB)
            }
            if (ay >= "0" && ay <= "9") {
                u(this, true);
                this.focus();
                ax.value = ay;
                ax.focus();
                return an(aB)
            }
            var av = aq("mn"), az = aB.shiftKey?-1 : this.date.getMonth(), aw = 0, at;
            while (++aw < 12) {
                at = av[(az + aw)%12].toLowerCase();
                if (at.indexOf(ay) == 0) {
                    var au = new Date(this.date);
                    au.setDate(1);
                    au.setMonth((az + aw)%12);
                    this.moveTo(au, true);
                    return an(aB)
                }
            }
            if (aD >= 37 && aD <= 40) {
                var au = this._lastHoverDate;
                if (!au&&!P.isEmpty()) {
                    au = aD < 39 ? P.getFirstDate() : P.getLastDate();
                    if (au < this._firstDateVisible || au > this._lastDateVisible) {
                        au = null
                    }
                }
                if (!au) {
                    au = aD < 39 ? this._lastDateVisible : this._firstDateVisible
                } else {
                    var aA = au;
                    au = B(au);
                    var az = 100;
                    while (az-->0) {
                        switch (aD) {
                        case 37:
                            au.setDate(au.getDate() - 1);
                            break;
                        case 38:
                            au.setDate(au.getDate() - 7);
                            break;
                        case 39:
                            au.setDate(au.getDate() + 1);
                            break;
                        case 40:
                            au.setDate(au.getDate() + 7);
                            break
                        }
                        if (!this.isDisabled(au)) {
                            break
                        }
                    }
                    au = V(au);
                    if (au < this._firstDateVisible || au > this._lastDateVisible) {
                        this.moveTo(au)
                    }
                }
                aj(this._getDateDiv(aA), M(this._getDateDiv(au), "DynarchCalendar-hover-date"));
                this._lastHoverDate = au;
                return an(aB)
            }
            if (aD == 13) {
                if (this._lastHoverDate) {
                    if (P.type == y.SEL_MULTIPLE && (aB.shiftKey || aB.ctrlKey)) {
                        if (aB.shiftKey && this._selRangeStart) {
                            P.clear(true);
                            P.selectRange(this._selRangeStart, this._lastHoverDate)
                        }
                        if (aB.ctrlKey) {
                            P.set(this._selRangeStart = this._lastHoverDate, true)
                        }
                    } else {
                        P.reset(this._selRangeStart = this._lastHoverDate)
                    }
                    return an(aB)
                }
            }
            if (aD == 27&&!this.args.cont) {
                this.hide()
            }
        }
    }
    l._getDateDiv = function(D) {
        var ar = null;
        if (D) {
            try {
                t(this.els.body, function(at) {
                    if (at.getAttribute("dyc-date") == D) {
                        throw ar = at
                    }
                })
            } catch (P) {}
        }
        return ar
    };
    function k(D, P) {
        return D.replace(/\$\{([^:\}]+)(:[^\}]+)?\}/g, function(av, au, at) {
            var aw = P[au], ar;
            if (at) {
                ar = at.substr(1).split(/\s*\|\s*/);
                aw = (aw >= ar.length ? ar[ar.length - 1] : ar[aw]).replace(/##?/g, function(ax) {
                    return ax.length == 2 ? "#" : aw
                })
            }
            return aw
        })
    }
    function aq(ar, P) {
        var D = q.__.data[ar];
        if (P && typeof D == "string") {
            D = k(D, P)
        }
        return D
    }(y.Selection = function(ar, P, D, at) {
        this.type = P;
        this.sel = ar instanceof Array ? ar : [ar];
        this.onChange = v(D, at);
        this.cal = at
    }).prototype = {
        get: function() {
            return this.type == y.SEL_SINGLE ? this.sel[0] : this.sel
        },
        isEmpty: function() {
            return this.sel.length == 0
        },
        set: function(P, D, ar) {
            var at = this.type == y.SEL_SINGLE;
            if (P instanceof Array) {
                this.sel = P;
                this.normalize();
                if (!ar) {
                    this.onChange(this)
                } else {
                    this.cal.refresh()
                }
            } else {
                P = V(P);
                if (at ||!this.isSelected(P)) {
                    at ? this.sel = [P] : this.sel.splice(this.findInsertPos(P), 0, P);
                    this.normalize();
                    if (!ar) {
                        this.onChange(this)
                    } else {
                        this.cal.refresh()
                    }
                } else {
                    if (D) {
                        this.unselect(P, ar)
                    }
                }
            }
        },
        reset: function() {
            this.sel = [];
            this.set.apply(this, arguments)
        },
        countDays: function() {
            var av = 0, D = this.sel, P = D.length, at, au, ar;
            while (--P >= 0) {
                at = D[P];
                if (at instanceof Array) {
                    au = B(at[0]);
                    ar = B(at[1]);
                    av += Math.round(Math.abs(ar.getTime() - au.getTime()) / 86400000)
                }
                ++av
            }
            return av
        },
        unselect: function(P, ar) {
            P = V(P);
            var at = false;
            for (var ay = this.sel, au = ay.length, D; --au >= 0;) {
                D = ay[au];
                if (D instanceof Array) {
                    if (P >= D[0] && P <= D[1]) {
                        var av = B(P), ax = av.getDate();
                        if (P == D[0]) {
                            av.setDate(ax + 1);
                            D[0] = V(av);
                            at = true
                        } else {
                            if (P == D[1]) {
                                av.setDate(ax - 1);
                                D[1] = V(av);
                                at = true
                            } else {
                                var aw = new Date(av);
                                aw.setDate(ax + 1);
                                av.setDate(ax - 1);
                                ay.splice(au + 1, 0, [V(aw), D[1]]);
                                D[1] = V(av);
                                at = true
                            }
                        }
                    }
                } else {
                    if (P == D) {
                        ay.splice(au, 1);
                        at = true
                    }
                }
            }
            if (at) {
                this.normalize();
                if (!ar) {
                    this.onChange(this)
                }
            }
        },
        normalize: function() {
            this.sel = this.sel.sort(function(ay, ax) {
                if (ay instanceof Array) {
                    ay = ay[0]
                }
                if (ax instanceof Array) {
                    ax = ax[0]
                }
                return ay - ax
            });
            for (var P = this.sel, ar = P.length, av, au; --ar >= 0;) {
                av = P[ar];
                if (av instanceof Array) {
                    if (av[0] > av[1]) {
                        P.splice(ar, 1);
                        continue
                    }
                    if (av[0] == av[1]) {
                        av = P[ar] = av[0]
                    }
                }
                if (au) {
                    var at = au, aw = av instanceof Array ? av[1]: av;
                    aw = B(aw);
                    aw.setDate(aw.getDate() + 1);
                    aw = V(aw);
                    if (aw >= at) {
                        var D = P[ar + 1];
                        if (av instanceof Array && D instanceof Array) {
                            av[1] = D[1];
                            P.splice(ar + 1, 1)
                        } else {
                            if (av instanceof Array) {
                                av[1] = au;
                                P.splice(ar + 1, 1)
                            } else {
                                if (D instanceof Array) {
                                    D[0] = av;
                                    P.splice(ar, 1)
                                } else {
                                    P[ar] = [av, D];
                                    P.splice(ar + 1, 1)
                                }
                            }
                        }
                    }
                }
                au = av instanceof Array ? av[0] : av
            }
        },
        findInsertPos: function(P) {
            for (var D = this.sel, ar = D.length, at; --ar >= 0;) {
                at = D[ar];
                if (at instanceof Array) {
                    at = at[0]
                }
                if (at <= P) {
                    break
                }
            }
            return ar + 1
        },
        clear: function(D) {
            this.sel = [];
            if (!D) {
                this.onChange(this)
            }
        },
        selectRange: function(ar, P) {
            ar = V(ar);
            P = V(P);
            if (ar > P) {
                var D = ar;
                ar = P;
                P = D
            }
            this.sel.push([ar, P]);
            this.normalize();
            this.onChange(this)
        },
        isSelected: function(D) {
            for (var P = this.sel.length, ar; --P >= 0;) {
                ar = this.sel[P];
                if (ar instanceof Array && D >= ar[0] && D <= ar[1] || D == ar) {
                    return true
                }
            }
            return false
        },
        getFirstDate: function() {
            var D = this.sel[0];
            if (D && D instanceof Array) {
                D = D[0]
            }
            return D
        },
        getLastDate: function() {
            if (this.sel.length > 0) {
                var D = this.sel[this.sel.length - 1];
                if (D && D instanceof Array) {
                    D = D[1]
                }
                return D
            }
        },
        print: function(ar, at) {
            var P = [], au = 0, aw, av = this.cal.getHours(), D = this.cal.getMinutes();
            if (!at) {
                at = " -> "
            }
            while (au < this.sel.length) {
                aw = this.sel[au++];
                if (aw instanceof Array) {
                    P.push(ab(B(aw[0], av, D), ar) + at + ab(B(aw[1], av, D), ar))
                } else {
                    P.push(ab(B(aw, av, D), ar))
                }
            }
            return P
        },
        getDates: function(P) {
            var D = [], ar = 0, au, at;
            while (ar < this.sel.length) {
                at = this.sel[ar++];
                if (at instanceof Array) {
                    au = B(at[0]);
                    at = at[1];
                    while (V(au) < at) {
                        D.push(P ? ab(au, P) : new Date(au));
                        au.setDate(au.getDate() + 1)
                    }
                } else {
                    au = B(at)
                }
                D.push(P ? ab(au, P) : au)
            }
            return D
        }
    };
    function a(P) {
        P = new Date(P.getFullYear(), P.getMonth(), P.getDate(), 12, 0, 0);
        var ar = P.getDay();
        P.setDate(P.getDate() - (ar + 6)%7 + 3);
        var D = P.valueOf();
        P.setMonth(0);
        P.setDate(4);
        return Math.round((D - P.valueOf()) / (7 * 86400000)) + 1
    }
    function C(D) {
        D = new Date(D.getFullYear(), D.getMonth(), D.getDate(), 0, 0, 0);
        var ar = new Date(D.getFullYear(), 0, 1, 12, 0, 0);
        var P = D - ar;
        return Math.floor(P / 86400000)
    }
    function V(D) {
        if (D instanceof Date) {
            return 10000 * D.getFullYear() + 100 * (D.getMonth() + 1) + D.getDate()
        }
        if (typeof D == "string") {
            return parseInt(D, 10)
        }
        return D
    }
    function B(ar, au, av, at, P) {
        if (!(ar instanceof Date)) {
            ar = parseInt(ar, 10);
            var aw = Math.floor(ar / 10000);
            ar = ar%10000;
            var D = Math.floor(ar / 100);
            ar = ar%100;
            ar = new Date(aw, D - 1, ar, au || 12, av || 0, at || 0, P || 0)
        }
        return ar
    }
    function af(aw, au, ar) {
        var av = aw.getFullYear(), ay = aw.getMonth(), P = aw.getDate(), at = au.getFullYear(), ax = au.getMonth(), D = au.getDate();
        return av < at?-3 : av > at ? 3 : ay < ax?-2 : ay > ax ? 2 : ar ? 0 : P < D?-1 : P > D ? 1 : 0
    }
    function ab(D, ax) {
        var P = D.getMonth(), aw = D.getDate(), ay = D.getFullYear(), az = a(D), aA = D.getDay(), aB = D.getHours(), ar = (aB >= 12), au = (ar) ? (aB - 12): aB, aD = C(D), at = D.getMinutes(), av = D.getSeconds(), aC = /%./g, aE;
        if (au === 0) {
            au = 12
        }
        aE = {
            "%a": aq("sdn")[aA],
            "%A": aq("dn")[aA],
            "%b": aq("smn")[P],
            "%B": aq("mn")[P],
            "%C": 1 + Math.floor(ay / 100),
            "%d": aw < 10 ? "0" + aw: aw,
            "%e": aw,
            "%H": aB < 10 ? "0" + aB: aB,
            "%I": au < 10 ? "0" + au: au,
            "%j": aD < 10 ? "00" + aD: aD < 100 ? "0" + aD: aD,
            "%k": aB,
            "%l": au,
            "%m": P < 9 ? "0" + (1 + P): 1 + P,
            "%o": 1 + P,
            "%M": at < 10 ? "0" + at: at,
            "%n": "\n",
            "%p": ar ? "PM": "AM",
            "%P": ar ? "pm": "am",
            "%s": Math.floor(D.getTime() / 1000),
            "%S": av < 10 ? "0" + av: av,
            "%t": "\t",
            "%U": az < 10 ? "0" + az: az,
            "%W": az < 10 ? "0" + az: az,
            "%V": az < 10 ? "0" + az: az,
            "%u": aA + 1,
            "%w": aA,
            "%y": ("" + ay).substr(2, 2),
            "%Y": ay,
            "%%": "%"
        };
        return ax.replace(aC, function(aF) {
            return aE.hasOwnProperty(aF) ? aE[aF] : aF
        })
    }
    function Y(P) {
        if (P) {
            if (typeof P == "number") {
                return B(P)
            }
            if (!(P instanceof Date)) {
                var D = P.split(/-/);
                return new Date(parseInt(D[0], 10), parseInt(D[1], 10) - 1, parseInt(D[2], 10), 12, 0, 0, 0)
            }
        }
        return P
    }
    function A(ar) {
        ar = ar.toLowerCase();
        function P(at) {
            for (var au = at.length; --au >= 0;) {
                if (at[au].toLowerCase().indexOf(ar) == 0) {
                    return au
                }
            }
        }
        var D = P(aq("smn")) || P(aq("mn"));
        if (D != null) {
            D++
        }
        return D
    }
    y.parseDate = function(au, D, aw) {
        if (!/\S/.test(au)) {
            return ""
        }
        au = au.replace(/^\s+/, "").replace(/\s+$/, "");
        aw = aw || new Date();
        var aB = null, P = null, aD = null, av = null, ar = null, aC = null;
        var ay = au.match(/([0-9]{1,2}):([0-9]{1,2})(:[0-9]{1,2})?\s*(am|pm)?/i);
        if (ay) {
            av = parseInt(ay[1], 10);
            ar = parseInt(ay[2], 10);
            aC = ay[3] ? parseInt(ay[3].substr(1), 10) : 0;
            au = au.substring(0, ay.index) + au.substr(ay.index + ay[0].length);
            if (ay[4]) {
                if (ay[4].toLowerCase() == "pm" && av < 12) {
                    av += 12
                } else {
                    if (ay[4].toLowerCase() == "am" && av >= 12) {
                        av -= 12
                    }
                }
            }
        }
        var az = au.split(/\W+/);
        var ax = [];
        for (var at = 0; at < az.length; ++at) {
            var aA = az[at];
            if (/^[0-9]{4}$/.test(aA)) {
                aB = parseInt(aA, 10);
                if (!P&&!aD && D == null) {
                    D = true
                }
            } else {
                if (/^[0-9]{1,2}$/.test(aA)) {
                    aA = parseInt(aA, 10);
                    if (aA >= 60) {
                        aB = aA
                    } else {
                        if (aA >= 0 && aA <= 12) {
                            ax.push(aA)
                        } else {
                            if (aA >= 1 && aA <= 31) {
                                aD = aA
                            }
                        }
                    }
                } else {
                    P = A(aA)
                }
            }
        }
        if (ax.length >= 2) {
            if (D) {
                if (!P) {
                    P = ax.shift()
                }
                if (!aD) {
                    aD = ax.shift()
                }
            } else {
                if (!aD) {
                    aD = ax.shift()
                }
                if (!P) {
                    P = ax.shift()
                }
            }
        } else {
            if (ax.length == 1) {
                if (!aD) {
                    aD = ax.shift()
                } else {
                    if (!P) {
                        P = ax.shift()
                    }
                }
            }
        }
        if (!aB) {
            aB = ax.length > 0 ? ax.shift() : aw.getFullYear()
        }
        if (aB < 30) {
            aB += 2000
        } else {
            if (aB < 99) {
                aB += 1900
            }
        }
        if (!P) {
            P = aw.getMonth() + 1
        }
        return aB && P && aD ? new Date(aB, P - 1, aD, av, ar, aC) : null
    };
    function al(D, at, P, ar) {
        ar = {};
        for (P in at) {
            if (at.hasOwnProperty(P)) {
                ar[P] = at[P]
            }
        }
        for (P in D) {
            if (D.hasOwnProperty(P)) {
                ar[P] = D[P]
            }
        }
        return ar
    }
    function p(ar, au, at, D) {
        if (ar instanceof Array) {
            for (var P = ar.length; --P >= 0;) {
                p(ar[P], au, at, D)
            }
        } else {
            if (typeof au == "object") {
                for (var P in au) {
                    if (au.hasOwnProperty(P)) {
                        p(ar, P, au[P], at)
                    }
                }
            } else {
                if (ar.addEventListener) {
                    ar.addEventListener(au, at, j ? true : !!D)
                } else {
                    if (ar.attachEvent) {
                        ar.attachEvent("on" + au, at)
                    } else {
                        ar["on" + au] = at
                    }
                }
            }
        }
    }
    function d(ar, au, at, D) {
        if (ar instanceof Array) {
            for (var P = ar.length; --P >= 0;) {
                d(ar[P], au, at)
            }
        } else {
            if (typeof au == "object") {
                for (var P in au) {
                    if (au.hasOwnProperty(P)) {
                        d(ar, P, au[P], at)
                    }
                }
            } else {
                if (ar.removeEventListener) {
                    ar.removeEventListener(au, at, j ? true : !!D)
                } else {
                    if (ar.detachEvent) {
                        ar.detachEvent("on" + au, at)
                    } else {
                        ar["on" + au] = null
                    }
                }
            }
        }
    }
    function an(D) {
        D = D || window.event;
        if (j) {
            D.cancelBubble = true;
            D.returnValue = false
        } else {
            D.preventDefault();
            D.stopPropagation()
        }
        return false
    }
    function aj(au, at, av) {
        if (au) {
            var D = au.className.replace(/^\s+|\s+$/, "").split(/\x20/), P = [], ar;
            for (ar = D.length; ar > 0;) {
                if (D[--ar] != at) {
                    P.push(D[ar])
                }
            }
            if (av) {
                P.push(av)
            }
            au.className = P.join(" ")
        }
        return av
    }
    function M(P, D) {
        return aj(P, D, D)
    }
    function ae(at, ar, P) {
        if (ar instanceof Array) {
            for (var D = ar.length; --D >= 0;) {
                ae(at, ar[D], P)
            }
        } else {
            aj(ar, P, at ? P : null)
        }
        return at
    }
    function G(at, D, ar) {
        var P = null;
        if (document.createElementNS) {
            P = document.createElementNS("http://www.w3.org/1999/xhtml", at)
        } else {
            P = document.createElement(at)
        }
        if (D) {
            P.className = D
        }
        if (ar) {
            ar.appendChild(P)
        }
        return P
    }
    function b(au, av) {
        if (av == null) {
            av = 0
        }
        var D, at, P;
        try {
            D = Array.prototype.slice.call(au, av)
        } catch (ar) {
            D = new Array(au.length - av);
            for (at = av, P = 0; at < au.length; ++at, ++P) {
                D[P] = au[at]
            }
        }
        return D
    }
    function v(P, ar) {
        var D = b(arguments, 2);
        return (ar == undefined ? function() {
            return P.apply(this, D.concat(b(arguments)))
        } : function() {
            return P.apply(ar, D.concat(b(arguments)))
        })
    }
    function t(P, ar) {
        if (!ar(P)) {
            for (var D = P.firstChild; D; D = D.nextSibling) {
                if (D.nodeType == 1) {
                    t(D, ar)
                }
            }
        }
    }
    function ap(D, aw, ar) {
        D = al(D, {
            fps: 50,
            len: 15,
            onUpdate: ad,
            onStop: ad
        });
        if (j) {
            D.len = Math.round(D.len / 2)
        }
        function at(aA, az, ax, ay) {
            return ay ? ax + aA * (az - ax) : az + aA * (ax - az)
        }
        function av() {
            if (aw) {
                P()
            }
            ar = 0;
            aw = setInterval(au, 1000 / D.fps)
        }
        function P() {
            if (aw) {
                clearInterval(aw);
                aw = null
            }
            D.onStop(ar / D.len, at)
        }
        function au() {
            var ax = D.len;
            D.onUpdate(ar / ax, at);
            if (ar == ax) {
                P()
            }
            ++ar
        }
        av();
        return {
            start: av,
            stop: P,
            update: au,
            args: D,
            map: at
        }
    }
    var Z = {
        elastic_b: function(D) {
            return 1 - Math.cos( - D * 5.5 * Math.PI) / Math.pow(2, 7 * D)
        },
        magnetic: function(D) {
            return 1 - Math.cos(D * D * D * 10.5 * Math.PI) / Math.exp(4 * D)
        },
        accel_b: function(D) {
            D = 1 - D;
            return 1 - D * D * D * D
        },
        accel_a: function(D) {
            return D * D * D
        },
        accel_ab: function(D) {
            D = 1 - D;
            return 1 - Math.sin(D * D * Math.PI / 2)
        },
        accel_ab2: function(D) {
            return (D/=0.5) < 1 ? 1 / 2 * D * D : - 1 / 2 * ((--D) * (D - 2) - 1)
        },
        brakes: function(D) {
            D = 1 - D;
            return 1 - Math.sin(D * D * Math.PI)
        },
        shake: function(D) {
            return D < 0.5?-Math.cos(D * 11 * Math.PI) * D * D : (D = 1 - D, Math.cos(D * 11 * Math.PI) * D * D)
        }
    };
    function am(D, P) {
        if (P === "") {
            j ? D.style.filter = "" : D.style.opacity = ""
        } else {
            if (P != null) {
                j ? D.style.filter = "alpha(opacity=" + P * 100 + ")" : D.style.opacity = P
            } else {
                if (!j) {
                    P = parseFloat(D.style.opacity)
                } else {
                    if (/alpha\(opacity=([0-9.])+\)/.test(D.style.opacity)) {
                        P = parseFloat(RegExp.$1) / 100
                    }
                }
            }
        }
        return P
    }
    function O(ar, D) {
        var P = ar.style;
        if (D != null) {
            P.display = D ? "" : "none"
        }
        return P.display != "none"
    }
    function f(P, ar) {
        var D = j ? P.clientX + document.body.scrollLeft: P.pageX;
        var at = j ? P.clientY + document.body.scrollTop: P.pageY;
        if (ar) {
            D -= ar.x;
            at -= ar.y
        }
        return {
            x: D,
            y: at
        }
    }
    function J(au) {
        var D = 0, at = 0, ar = /^div$/i.test(au.tagName), av, P;
        if (ar && au.scrollLeft) {
            D = au.scrollLeft
        }
        if (ar && au.scrollTop) {
            at = au.scrollTop
        }
        av = {
            x: au.offsetLeft - D,
            y: au.offsetTop - at
        };
        if (au.offsetParent) {
            P = J(au.offsetParent);
            av.x += P.x;
            av.y += P.y
        }
        return av
    }
    function X() {
        var P = document.documentElement, D = document.body;
        return {
            x: P.scrollLeft || D.scrollLeft,
            y: P.scrollTop || D.scrollTop,
            w: P.clientWidth || window.innerWidth || D.clientWidth,
            h: P.clientHeight || window.innerHeight || D.clientHeight
        }
    }
    function m(D, ar, P) {
        for (P = 0; P < D.length; ++P) {
            ar(D[P])
        }
    }
    var ad = new Function();
    function W(D) {
        if (typeof D == "string") {
            D = document.getElementById(D)
        }
        return D
    }
    return y
})();
CalendarDB.LANG("en", "English", {
    fdow: 1,
    goToday: "Go Today",
    today: "Today",
    wk: "wk",
    weekend: "0,6",
    AM: "am",
    PM: "pm",
    mn: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    smn: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    dn: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    sdn: ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
});
var Article = {
    SendingAjax: false,
    SettingsArticleListOnChange: function(ctl, ctlPubDateId, ctlKillDateId)
    {
        var ctlPubDate = $(ctlPubDateId);
        var ctlKillDate = $(ctlKillDateId);
        var selectedArtInfo = ctl.options[ctl.selectedIndex].value;
        if (selectedArtInfo.length > 0)
        {
            ctlPubDate.value = selectedArtInfo.split(',')[1];
            ctlKillDate.value = selectedArtInfo.split(',')[2];
        } else
        {
            ctlPubDate.value = "";
            ctlKillDate.value = "";
        }
    },
    SettingsEditArticle: function(ctl, ctlAttHdnOpsId, ctlArticleListId, ctlpubDateId, ctlkillDateId, CurrentArtsDivId, articleId, pubDate, killDate)
    {
        var ctlAttHdnOp = $(ctlAttHdnOpsId);
        var ctlArticleList = $(ctlArticleListId);
        var ctlpubDate = $(ctlpubDateId);
        var ctlkillDate = $(ctlkillDateId);
        var ctlCurrentArtsDiv = $(CurrentArtsDivId);
        var articlesRows = ctlCurrentArtsDiv.childNodes;
        for (i = 0; i < articlesRows.length; i++)
        {
            articlesRows[i].childNodes[2].style.fontWeight = "";
            articlesRows[i].childNodes[2].style.textDecoration = "";
        }
        ctl.parentNode.parentNode.childNodes[2].style.fontWeight = "bold";
        ctlAttHdnOp.value = "Edit," + articleId;
        ctlArticleList.disabled = true;
        ctlpubDate.value = pubDate;
        ctlkillDate.value = killDate;
    },
    SettingsDeleteArticle: function(ctl, ctlAttHdnOpsId, ctlArticleListId, ctlpubDateId, ctlkillDateId, articleId)
    {
        var ctlAttHdnOp = $(ctlAttHdnOpsId);
        var ctlpubDate = $(ctlpubDateId);
        var ctlkillDate = $(ctlkillDateId);
        var ctlArticleList = $(ctlArticleListId);
        ctlpubDate.value = "";
        ctlkillDate.value = "";
        var articlesRows = ctl.parentNode.parentNode.parentNode.childNodes;
        var i;
        for (i = 0; i < articlesRows.length; i++)
        {
            articlesRows[i].childNodes[2].style.textDecoration = "";
            articlesRows[i].childNodes[2].style.fontWeight = "";
        }
        ctl.parentNode.parentNode.childNodes[2].style.textDecoration = "line-through";
        ctl.parentNode.parentNode.childNodes[2].style.fontWeight = "bold";
        ctlAttHdnOp.value = "Del," + articleId;
        ctlArticleList.disabled = true;
    }
}
var AllArticlesList = Class.create({
    SendingAjax: false,
    ModuleUniqueId: 0,
    ArticlesToShow: 0,
    CurrentPage: 1,
    PagesToShow: 0,
    CurrentBlock: 1,
    ModuleIdentity: '',
    DivArticlesId: '',
    DivPagesId: '',
    GoToPage: function(pObj, pArticlesObj, pPageToGo)
    {
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        new Ajax.Request('/ajax/?a=[DonbestCommon]Articles_ChangeArticlesListPage', {
            method: 'post',
            parameters: {
                articlesToShow: pArticlesObj.ArticlesToShow,
                pageToGo: pPageToGo
            },
            requestHeaders: {
                Accept: 'application/json'
            },
            onSuccess: function(result) {
                this.DisplayArticles(pArticlesObj, pPageToGo, result.responseJSON);
            }.bind(this),
            onComplete: function(t) {
                this.SendingAjax = false;
            }.bind(this)
        });
        return false;
    },
    DisplayArticles: function(pArticlesObj, pPageToGo, articlesList)
    {
        var ArticlesContainer = document.createElement("div");
        var PagesContainer = document.createElement("div");
        var allArticlesContainer = document.createElement("div");
        pArticlesObj.CurrentPage = pPageToGo;
        var articlesLength = articlesList.PagedArticlesList.Total;
        var articles = articlesList.PagedArticlesList.Articles;
        var i;
        for (i = 0; i < articles.length; i++)
        {
            var artContainer = document.createElement("div");
            var headerContainer = document.createElement("div");
            var titleContainer = document.createElement("div");
            var titleLnk = document.createElement("a");
            var dateContainer = document.createElement("div");
            var reviewContainer = document.createElement("div");
            var article = articles[i].ArticleInfo;
            headerContainer.className = "all_article_header";
            titleLnk.href = article.Link;
            titleLnk.innerHTML = article.Title;
            titleContainer.appendChild(titleLnk);
            titleContainer.className = "all_article_header_title";
            var date = new Date(article.PublicationDate);
            dateContainer.innerHTML = date.getMonth() + '/' + date.getDate() + '/' + date.getFullYear();
            dateContainer.className = "all_article_header_date";
            headerContainer.appendChild(titleContainer);
            headerContainer.appendChild(dateContainer);
            reviewContainer.innerHTML = article.Blurb + "..." + "<a href=\"" + article.Link + "\" class=\"more\" >Read this article</a>"
            reviewContainer.className = "all_article_review";
            artContainer.appendChild(headerContainer);
            artContainer.appendChild(reviewContainer);
            artContainer.className = "all_article";
            allArticlesContainer.appendChild(artContainer);
            artContainer = null;
            headerContainer = null;
            titleContainer = null;
            titleLnk = null;
            dateContainer = null;
            reviewContainer = null;
        }
        ArticlesContainer.appendChild(allArticlesContainer);
        var DivPage;
        var AnchorPage;
        var totalPages = Math.ceil(articlesLength / pArticlesObj.ArticlesToShow);
        var currentPages = totalPages > pArticlesObj.PagesToShow ? pArticlesObj.PagesToShow: totalPages;
        if (pArticlesObj.CurrentBlock * currentPages < pPageToGo)
        {
            pArticlesObj.CurrentBlock += 1;
        }
        if (pArticlesObj.CurrentBlock * currentPages - currentPages == pPageToGo)
        {
            pArticlesObj.CurrentBlock -= 1;
        }
        if (pArticlesObj.CurrentBlock > 1)
        {
            DivPage = document.createElement("div");
            AnchorPage = document.createElement("a");
            AnchorPage.href = "#";
            AnchorPage.innerHTML = "&laquo;";
            DivPage.className = "control_page";
            var prevPage = ((pArticlesObj.CurrentBlock * currentPages) - (currentPages - 1)) - 1;
            var pagerFunction = "return " + pArticlesObj.ModuleIdentity + ".GoToPage(this, " + pArticlesObj.ModuleIdentity + ", " + prevPage + ");";
            if (AnchorPage.attachEvent)
                AnchorPage.attachEvent("onclick", new Function(pagerFunction));
            else
                AnchorPage.setAttribute("onclick", pagerFunction);
            DivPage.appendChild(AnchorPage);
            PagesContainer.appendChild(DivPage);
            AnchorPage = null;
            DivPage = null;
        }
        for (var iPage = 1; iPage <= currentPages; iPage++)
        {
            var pageToDisplay = (pArticlesObj.CurrentBlock * currentPages) - (currentPages - iPage);
            if (articlesLength <= (pageToDisplay * pArticlesObj.ArticlesToShow) - pArticlesObj.ArticlesToShow)
                break;
            DivPage = document.createElement("div");
            AnchorPage = document.createElement("a");
            AnchorPage.href = "#";
            AnchorPage.innerHTML = pageToDisplay;
            if (pPageToGo == pageToDisplay)
                DivPage.className = "control_page_selected";
            else
                DivPage.className = "control_page";
            var pagerFunction = "return " + pArticlesObj.ModuleIdentity + ".GoToPage(this, " + pArticlesObj.ModuleIdentity + ", " + pageToDisplay + ");";
            if (AnchorPage.attachEvent)
                AnchorPage.attachEvent("onclick", new Function(pagerFunction));
            else
                AnchorPage.setAttribute("onclick", pagerFunction);
            DivPage.appendChild(AnchorPage);
            PagesContainer.appendChild(DivPage);
            AnchorPage = null;
            DivPage = null;
        }
        if (totalPages > (currentPages * pArticlesObj.CurrentBlock))
        {
            DivPage = document.createElement("div");
            AnchorPage = document.createElement("a");
            AnchorPage.href = "#";
            AnchorPage.innerHTML = "&raquo;";
            DivPage.className = "control_page";
            var nextPage = (pArticlesObj.CurrentBlock * currentPages) + 1;
            var pagerFunction = "return " + pArticlesObj.ModuleIdentity + ".GoToPage(this, " + pArticlesObj.ModuleIdentity + ", " + nextPage + ");";
            if (AnchorPage.attachEvent)
                AnchorPage.attachEvent("onclick", new Function(pagerFunction));
            else
                AnchorPage.setAttribute("onclick", pagerFunction);
            DivPage.appendChild(AnchorPage);
            PagesContainer.appendChild(DivPage);
            AnchorPage = null;
            DivPage = null;
        }
        var Module = $('moduleData' + pArticlesObj.ModuleUniqueId);
        var _DivArticlesCtl = Module.select('[id=' + pArticlesObj.DivArticlesId + ']').first();
        var _DivPages = Module.select('[id=' + pArticlesObj.DivPagesId + ']').first();
        _DivArticlesCtl.innerHTML = '';
        _DivPages.innerHTML = '';
        _DivArticlesCtl.appendChild(ArticlesContainer);
        _DivPages.appendChild(PagesContainer);
        _DivArticlesCtl = null;
        _DivPages = null;
    }
});
var AutoRefresh = Class.create({
    initialize: function(options) {
        this.options = {
            objId: '',
            obj: null,
            status: this._GetStatus(),
            seconds: 10,
            OnChangeStatus: null,
            refreshTO: 0,
            currentPage: - 1
        }
        Object.extend(this.options, options || {});
        if (!this.options.objId.empty())
            this.options.obj = $(this.options.objId);
        if (this.options.obj != null) {
            Event.observe(this.options.obj, 'click', function() {
                this.Toggle();
            }.bind(this));
            if (typeof(this.options.OnChangeStatus) == 'function')
                this.options.OnChangeStatus();
            if (this.options.status)
                this.StartRefresh();
        }
    }
});
AutoRefresh.Status = {
    OFF: false,
    ON: true
};
AutoRefresh.prototype._GetStatus = function() {
    var current = AutoRefresh.Status.OFF;
    var params = window.location.href.toQueryParams();
    if (params.autorefresh != null && params.autorefresh == 'true') {
        return AutoRefresh.Status.ON;
    } else {
        if (isdefined('DEFAULT_AUTO_REFRESH'))
        {
            if (DEFAULT_AUTO_REFRESH.toLowerCase() == "off")
            {
                return AutoRefresh.Status.OFF;
            } else {
                return AutoRefresh.Status.ON;
            }
        } else {
            return AutoRefresh.Status.OFF;
        }
    }
};
AutoRefresh.prototype.Toggle = function() {
    this.options.status = this.options.status == AutoRefresh.Status.OFF ? AutoRefresh.Status.ON : AutoRefresh.Status.OFF;
    if (typeof(this.options.OnChangeStatus) == 'function') {
        this.options.OnChangeStatus();
    }
    if (!this.options.status)
        this.StopRefresh();
    if (this.options.status)
        this.StartRefresh();
}
AutoRefresh.prototype.StopRefresh = function() {
    clearTimeout(this.options.refreshTO);
}
AutoRefresh.prototype.StartRefresh = function() {
    if (this.options.seconds < 5)
        this.options.seconds = 5;
    this.options.refreshTO = setTimeout(function() {
        var params = window.location.href.toQueryParams();
        params.autorefresh = this.options.status;
        var s = new StringBuilder();
        for (k in params)
            if (params[k] != null)
            {
                if (k != "page")
                    s.AppendFormat("#{0}=#{1}", {
                        0: k,
                        1: params[k]
                    });
            }
        if (this.options.currentPage!=-1)
        {
            s.AppendFormat("#{0}=#{1}", {
                0: "&page",
                1: this.options.currentPage
            });
        }
        var w = {
            host: window.location.hostname,
            path: window.location.pathname,
            query: s.ToString('&')
        };
        path = 'http://#{host}#{path}?#{query}'.interpolate(w);
        window.location = path;
    }.bind(this), this.options.seconds * 1000);
}
function isdefined(variable) {
    return (typeof(window[variable]) == "undefined") ? false : true;
}
var BooksModule = {
    Load: function(obj) {
        $('book-list3c').select('tr.sb-row').each(function(tr, i) {
            var info = obj[i + 1];
            if (info == null)
                return;
            tr.down('td a.book-title').update(info.name).writeAttribute({
                href: info.link
            }).className = info.classes;
            tr.next('tr').down('td').update(info.review);
        });
    }
}
var DonBest3 = {
    ToggleBookInfo: function(obj) {
        if (obj.visible()) {
            obj.hide();
            return;
        }
        obj.up('table').select('tr.bookReviewInfo').each(function(d, i) {
            d.hide();
        });
        obj.show();
    }
};
var Calendar = function(vars) {
    var g = this;
    this.vars = vars;
    this.holder = null;
    this.startDate = g.vars.startDate ? g.vars.startDate : new Date();
    this.calendarClass = g.vars.calendarClass ? g.vars.calendarClass : '';
    this.currentDate = g.startDate;
    this.cells = Array();
    this.onNext = g.vars.onNext ? g.vars.onNext : null;
    this.onPrevious = g.vars.onPrevious ? g.vars.onPrevious : null;
    this.monthHeader = null;
    this.month = null;
    this.init = function() {
        g.holder = $(g.vars.holderId);
        g.makeCalendar(g.holder, g.startDate);
        if (g.vars.onInit)
            g.vars.onInit(g);
    }
    this.reInit = function() {
        g.holder = $(g.vars.holderId);
        g.makeCalendar(g.holder, g.currentDate);
    }
    this.clear = function() {
        g.holder.update('');
        g.cells = null;
        g.cells = new Array();
    }
    this.Next = function(d1) {
        g.clear();
        var thisDate = new Date();
        thisDate.setFullYear(g.currentDate.getFullYear(), g.currentDate.getMonth() + 1, 1);
        g.currentDate = thisDate;
        g.makeCalendar(g.holder, thisDate);
        if (g.onNext)
            g.onNext(g.cells, new g.Month(g.currentDate.getMonth(), g.currentDate.getFullYear()));
    }
    this.Previous = function() {
        g.clear();
        var thisDate = new Date();
        thisDate.setFullYear(g.currentDate.getFullYear(), g.currentDate.getMonth() - 1, 1);
        g.currentDate = thisDate;
        g.makeCalendar(g.holder, thisDate);
        if (g.onPrevious)
            g.onPrevious(g.cells, new g.Month(g.currentDate.getMonth(), g.currentDate.getFullYear()));
    }
    this.Create = function() {
        g.holder.innerHTML = '';
        g.makeCalendar(g.holder, g.currentDate);
    }
    this.makeCalendar = function(obj, d) {
        var c = this;
        this.d = d;
        this.obj = obj;
        c.month = new g.Month(d.getMonth(), d.getFullYear());
        g.month = c.month;
        var table = new Element('table', {
            cellspacing: 0,
            cellpadding: 0,
            className: g.calendarClass
        });
        if (Prototype.Browser.IE)
            var tbody = new Element('tbody');
        var tr = new Element('tr');
        tr.className = g.calendarClass + "_header";
        var td = new Element('td');
        if (g.vars.showPrevious) {
            td.innerHTML = (g.vars.nextText ? g.vars.prevText : '&lt;&lt;');
            td.className = g.calendarClass + '_prev';
            Event.observe(td, "click", function() {
                g.Previous(d);
            });
        } else {
            td.innerHTML = '&nbsp;';
        }
        tr.appendChild(td);
        var td1 = new Element('td', {
            colSpan: 5,
            className: g.calendarClass + '_month'
        });
        td1.innerHTML = c.month.Name + "&nbsp;" + c.month.year;
        if (g.vars.onClickMonth) {
            Event.observe(td1, "click", function() {
                g.vars.onClickMonth(td1, c.month);
            });
        }
        g.monthHeader = td1;
        tr.appendChild(td1);
        td = new Element('td');
        if (g.vars.showNext) {
            Event.observe(td, "click", function() {
                g.Next(d);
            });
            td.innerHTML = (g.vars.nextText ? g.vars.nextText : '&gt;&gt;');
            td.className = g.calendarClass + '_next';
        } else
            td.innerHTML = '&nbsp;';
        tr.appendChild(td);
        if (Prototype.Browser.IE)
            tbody.appendChild(tr);
        else
            table.appendChild(tr);
        tr = new Element('tr');
        for (i = 0; i < g.WeekDays.length; i++) {
            td = new Element('td', {
                className: g.calendarClass + '_days'
            });
            td.innerHTML = g.WeekDays[i];
            tr.appendChild(td);
        }
        if (Prototype.Browser.IE)
            tbody.appendChild(tr);
        else
            table.appendChild(tr);
        var weeks = c.month.Weeks();
        for (i = 0; i < weeks.length; i++) {
            tr = new Element('tr');
            for (k = 0; k < weeks[i].length; k++) {
                g.cells.push(weeks[i][k]);
                tr.appendChild(weeks[i][k]);
            }
            if (Prototype.Browser.IE)
                tbody.appendChild(tr);
            else
                table.appendChild(tr);
        }
        if (Prototype.Browser.IE)
            table.appendChild(tbody);
        obj.appendChild(table);
        if (g.vars.onCreate)
            g.vars.onCreate(g);
        table = null;
        tr = null;
        td = null;
    }
    this.Month = function(id, year) {
        var m = this;
        this.id = id;
        this.year = year;
        this.Days = id != 1 ? g.MonthDays[m.id] : m.year%4 == 0 ? g.MonthDays[m.id] + 1 : g.MonthDays[m.id];
        this.date = new Date();
        this.date.setFullYear(m.year, m.id, 1);
        this.lastDate = new Date();
        this.lastDate.setFullYear(m.year, m.id, m.Days);
        this.Name = g.MonthNames[m.id];
        this.ShortName = g.MonthNames[m.id].substr(0, 3);
        this.StartDay = m.date.getDay();
        this.firstFullWeek = m.StartDay == 0;
        this.lastFullWeek = (m.StartDay + m.Days)%7 - 1==-1;
        this.WeekCount = Math.ceil((m.Days + m.StartDay) / 7);
        this.firstCell = null, this.lastCell = null;
        this.Weeks = function() {
            var weeks = Array();
            if (m.firstFullWeek) {
                weeks.push(m.Previous().LastWeek('last'));
                weeks.push(m.FirstWeek('this'));
            } else
                weeks.push(m.Previous().LastWeek('last').concat(m.FirstWeek('this')));
            for (j = 1; j < m.WeekCount - 1; j++)
                weeks.push(m.GetWeek(j));
            if (m.lastFullWeek) {
                weeks.push(m.LastWeek('this'));
                weeks.push(m.Next().FirstWeek('next'));
            } else
                weeks.push(m.LastWeek('this').concat(m.Next().FirstWeek('next')));
            var cnt = m.WeekCount + (m.lastFullWeek ? 1 : 0) + (m.firstFullWeek ? 1 : 0)
            if (g.vars.onDateSelect) {
                for (j = 0; j < cnt; j++) {
                    for (k = 0; k < weeks[j].length; k++) {
                        Event.observe(weeks[j][k], "click", function() {
                            var ret = {
                                id: this.id,
                                date: new Date(this.getAttribute("value")),
                                cell: this,
                                otherMonth: (this.className.indexOf('last') >= 0 || this.className.indexOf('next') >= 0)
                            };
                            if (g.vars.onDateSelect)
                                g.vars.onDateSelect(ret);
                        });
                    }
                }
            }
            return weeks;
        }
        this.GetWeek = function(_int) {
            var gw = this;
            var days = Array();
            gw.dated = m.id + 1 + '/{0}/' + m.year;
            if (_int == 0)
                return m.FirstWeek('this');
            if (_int == m.WeekCount)
                return m.LastWeek('this');
            var start = 7 - m.StartDay + 1;
            var s = ""
            for (i = 0; i < 7; i++) {
                var val = (7 * (_int - 1) + start + i)
                var tdate = new Date(gw.dated.replace('{0}', val));
                var fd = dateFormat(tdate, "mm/dd/yyyy");
                var d = new Element('td', {
                    id: "cal_" + fd.replace(/\//g, ""),
                    className: g.calendarClass + '_thisMonth',
                    value: fd,
                    month: 'this'
                });
                if (m.MatchDate(tdate, g.startDate))
                    $(d).addClassName(g.calendarClass + '_today');
                d.appendChild(new Element('span').update(val));
                days.push(d);
            }
            return days;
        }
        this.MatchDate = function(d1, d2) {
            var d01, d02;
            d01 = "#{0}/#{1}/#{2}".interpolate({
                0: d1.getMonth() + 1,
                1: d1.getDate(),
                2: d1.getFullYear()
            });
            d02 = "#{0}/#{1}/#{2}".interpolate({
                0: d2.getMonth() + 1,
                1: d2.getDate(),
                2: d2.getFullYear()
            });
            if (d01 == d02)
                return true;
            return false;
        }
        this.LastWeek = function(c) {
            var lw = this;
            var days = Array();
            lw.dated = m.id + 1 + '/{0}/' + m.year;
            var day = (m.StartDay + m.Days)%7 - 1;
            day = day==-1 ? 6 : day;
            var s = "";
            for (i = m.Days - day; i <= m.Days; i++) {
                var tdate = new Date(lw.dated.replace("{0}", i));
                var fd = dateFormat(tdate, "mm/dd/yyyy");
                var d = new Element('td', {
                    id: "cal_" + fd.replace(/\//g, ""),
                    className: g.calendarClass + '_' + c + 'Month',
                    value: fd,
                    month: c
                });
                d.appendChild(new Element('span').update(i));
                if (m.MatchDate(tdate, g.startDate))
                    $(d).addClassName(g.calendarClass + '_today');
                days.push(d);
                if (i == m.Days)
                    m.lastCell = d;
            }
            return days;
        }
        this.FirstWeek = function(c) {
            var fw = this;
            var days = Array();
            fw.dated = m.id + 1 + '/{0}/' + m.year;
            var day = 7 - m.StartDay;
            var s = "";
            for (i = 1; i <= day; i++) {
                var tdate = new Date(fw.dated.replace('{0}', i));
                var fd = dateFormat(tdate, "mm/dd/yyyy");
                var d = new Element('td', {
                    id: "cal_" + fd.replace(/\//g, ""),
                    className: g.calendarClass + '_' + c + 'Month',
                    value: fd,
                    month: c
                });
                d.appendChild(new Element('span').update(i));
                if (m.MatchDate(tdate, g.startDate))
                    $(d).addClassName(g.calendarClass + '_today');
                days.push(d);
                if (i == 1)
                    m.firstCell = d;
            }
            return days;
        }
        this.Previous = function() {
            var pid = m.id - 1 < 0 ? 11: m.id - 1;
            var pyear = pid == 11 ? m.year - 1: m.year;
            return new g.Month(pid, pyear);
        }
        this.Next = function() {
            var pid = m.id + 1 > 11 ? 0: m.id + 1;
            var pyear = pid == 0 ? m.year + 1: m.year;
            return new g.Month(pid, pyear);
        }
    }
    this.WeekDays = g.vars.WeekDays ? g.vars.WeekDays : new Array('S', 'M', 'T', 'W', 'T', 'F', 'S');
    this.MonthDays = new Array(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31);
    this.MonthNames = new Array('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December');
    if (g.vars.autostart || g.vars.autostart == null)
        g.init();
}
var PopupCalendar = function(vars) {
    var g = this;
    this.vars = vars;
    this.textBox = $(g.vars.textId);
    this.trigger = $(g.vars.triggerId);
    this.holder = $(g.vars.holderId);
    this.calendar = null;
    this.to = 0;
    this.DisableDate = new Date().setDate(new Date().getDate() - 1);
    this.Show = function() {
        var pos = g.textBox.cumulativeOffset();
        g.holder.setStyle({
            position: 'absolute',
            left: pos.left + 'px',
            top: pos.top + g.textBox.getHeight() + 'px'
        });
        clearTimeout(g.to);
        if (!g.holder.visible())
            if (Prototype.Browser.IE)
                g.holder.setStyle({
                    display: 'block'
                });
            else
                new Effect.toggle(g.holder, 'blind', {
                    duration: .3
                });
    }
    this.Hide = function() {
        g.to = setTimeout(function() {
            if (g.holder.visible())
                new Effect.toggle(g.holder, 'blind', {
                    duration: .2
                });
        }, 200);
    }
    this.init = function() {
        Event.observe(g.textBox, "focus", function() {
            g.Show();
        });
        Event.observe(g.textBox, "blur", function() {
            g.Hide();
        });
        Event.observe(g.holder, "mouseover", function() {
            clearTimeout(g.to);
        });
        Event.observe(g.holder, "click", function() {
            clearTimeout(g.to);
        });
        Event.observe(g.trigger, "click", function() {
            if (g.holder.visible())
                g.Hide();
            else 
                g.Show();
        });
        Event.observe(window, "load", function() {
            document.body.appendChild(g.holder);
        });
        if (!g.vars['disablePreviousDates']) {
            g.DisableDate = false;
        }
        g.CreateCalendar();
        g.PaintCalendar();
    }
    this.CreateCalendar = function() {
        g.calendar = null;
        g.holder.update('');
        g.calendar = new Calendar({
            holderId: g.holder.id,
            startDate: new Date(g.vars.startDate),
            calendarClass: 'popup_calendar',
            showPrevious: true,
            showNext: true,
            nextText: '►',
            prevText: '◄',
            onDateSelect: function(date) {
                g.updateText(date);
                g.holder.setStyle({
                    display: 'none'
                });
            },
            onCreate: function() {
                if (g.calendar)
                    g.PaintCalendar();
                var button = new Element('span', {
                    'class': 'popup_today'
                }).update('Today');
                Event.observe(button, "click", function() {
                    g.textBox.setValue(dateFormat(new Date(), "mm/dd/yyyy"));
                });
                var todayHolder = new Element('div', {
                    'class': 'popup_todayHolder'
                })
                todayHolder.appendChild(button);
                button = new Element('span', {
                    'class': 'popup_today',
                    style: 'padding:2px 17px 2px 18px;margin-left:1px;'
                }).update('Close');
                Event.observe(button, 'click', function() {
                    g.textBox.blur();
                    g.Hide();
                });
                todayHolder.appendChild(button);
                g.holder.appendChild(todayHolder);
            },
            onNext: function() {
                g.textBox.focus();
            },
            onPrev: function() {
                g.textBox.focus();
            }
        });
    }
    this.PaintCalendar = function() {
        if (g.DisableDate == null)
            return;
        $(g.calendar.cells).each(function(cell) {
            var d = new Date(cell.getAttribute('value'));
            if (d <= g.DisableDate)
                cell.addClassName('popup_cellDisabled');
            else
                cell.removeClassName('popup_cellDisabled');
        });
    }
    this.updateText = function(date) {
        if (g.DisableDate)
            if (date.date <= g.DisableDate)
                return;
        g.textBox.setValue(dateFormat(date.date, "mm/dd/yyyy"));
        if (g.vars.endDateId != '') {
            eval('#{0}.DisableDate = new Date(#{1});#{0}.PaintCalendar();'.interpolate({
                0: g.vars.endDateId,
                1: date.date.getTime()
            }));
        }
    }
    this.init();
}
var Comment = {
    SendingAjax: false,
    EditValue: '',
    AddComment: function(obj) {
        if (this.SendingAjax)
            return;
        x = SBRForum.FormValues(obj);
        this.SendingAjax = true;
        new Ajax.Request('/ajax/?a=Comment_AddComment', {
            method: 'post',
            parameters: {
                input: x.serialize
            },
            onSuccess: function(t) {
                Comment.ClearInputs(obj);
                Comment.AddCommentToList(t.responseText.evalJSON(false).message);
            },
            onComplete: function(t) {
                Comment.SendingAjax = false;
            }
        });
    },
    ShowEditComment: function(obj, id) {
        obj = $(obj).ancestors()[1].select('.textAdmin')[0];
        this.EditValue = obj.innerHTML;
        obj.update('');
        var input = new Element('textarea', {
            'class': 'editBox'
        });
        Event.observe(input, 'onclick', function(e) {
            Event.stop(e);
        });
        Event.observe(input, 'blur', function() {
            Comment.HideEditComment(input, id);
        });
        input.value = this.EditValue;
        obj.parentNode.appendChild(input);
        input.focus();
    },
    HideEditComment: function(obj, id) {
        var input = $(obj);
        obj = $(obj).parentNode;
        this.SendingAjax = true;
        new Ajax.Request('/ajax/?a=Comment_EditComment', {
            method: 'post',
            parameters: {
                'id': id,
                'text': input.value
            },
            onSuccess: function(t) {
                Comment.EditValue = input.value;
                obj.removeChild(input);
                $(obj).select('.textAdmin')[0].update(Comment.EditValue);
            },
            onComplete: function(t) {
                Comment.SendingAjax = false;
                Comment.EditValue = '';
            }
        });
    },
    DeleteComment: function(obj, id) {
        obj = $(obj).ancestors()[2];
        this.SendingAjax = true;
        new Ajax.Request('/ajax/?a=Comment_DeleteComment', {
            method: 'post',
            parameters: {
                'id': id
            },
            onSuccess: function(t) {
                $(obj).hide();
            },
            onComplete: function(t) {
                Comment.SendingAjax = false;
            }
        });
    },
    ClearInputs: function(obj) {
        $(obj).select('#comment')[0].value = '';
        $(obj).select('#username')[0].value = '';
        $(obj).select('#avatar')[0].value = '';
    },
    AddCommentToList: function(message) {
        var cpy = $('newComment').cloneNode(true);
        $(cpy).select('.avatar img')[0].src = message.avatar;
        $(cpy).select('strong')[0].update(message.username)
        $(cpy).select('.commentDate')[0].update(message.date);
        $(cpy).select('.text')[0].update(message.text);
        $(cpy).show();
        if ($('comments').firstDescendant() != null)
            $('comments').insertBefore(cpy, $('comments').firstDescendant());
        else
            $('comments').appendChild(cpy);
    }
}
var ContactForm = Class.create({
    Options: {},
    SendingAjax: false,
    initialize: function(options) {
        this.options = {
            _Name: ''
        };
        Object.extend(this.Options, options || {});
    },
    Cache: {},
    GetFrontPage: function(obj, name, email, subject, body, onSend) {
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        Object.extend(this.Options, {
            name: name,
            email: email,
            subject: subject,
            body: body
        }
        || {});
        new Ajax.Request('/ajax/?a=[CMS]ContactForm_SendMail', {
            method: 'post',
            parameters: this.Options,
            onSuccess: function(t) {
                var data = t.responseJSON;
                obj = $(obj);
                if (onSend != null)
                    onSend();
            }.bind(this),
            onComplete: function(t) {
                this.SendingAjax = false;
                $(obj).value = 'Send';
                $(obj).disabled = false;
                alert('Thank you for contacting us.');
            }.bind(this)
        });
    },
    SendMail: function(obj, clientId) {
        $(obj).value = 'Sending ...';
        $(obj).disabled = true;
        var doc = document;
        var name = doc.getElementById('txtName_' + clientId);
        var email = doc.getElementById('txtEmail_' + clientId);
        var subject = doc.getElementById('txtSubject_' + clientId);
        var body = doc.getElementById('txtBody_' + clientId);
        var msg = this.IsValid(name.value, email.value, subject.value, body.value);
        var div = doc.getElementById("msg_" + clientId);
        if (msg == '') {
            this.GetFrontPage(obj, name.value, "nobody@sbrmarketing.com", subject.value, "EMAIL: #{0}<br />#{1}".interpolate({
                0: email.value,
                1: body.value
            }), function() {
                this.Clean(name, email, subject, body, div);
            }.bind(this));
        } else {
            div.innerHTML = msg;
        }
    },
    IsValid: function(pName, pEmail, pSubject, pBody) {
        if (pName == null || pName == '')
            return 'Please submit a name';
        var reg = /^([A-Za-z0-9_\-\.])+\@([A-Za-z0-9_\-\.])+\.([A-Za-z]{2,4})$/;
        if (reg.test(pEmail) == false)
            return 'Invalid email address';
        if (pSubject == null || pSubject == '')
            return 'Please type a subject';
        if (pBody == null || pBody == '')
            return 'Please check the Body of your message';
        return "";
    },
    Clean: function(pName, pEmail, pSubject, pBody, pMsg) {
        pName.value = "";
        pEmail.value = "";
        pSubject.value = "";
        pBody.value = "";
        pMsg.innerHTML = "";
    }
});
var DBNewsbreaker = Class.create({
    Options: {},
    SendingAjax: false,
    initialize: function(options) {
        this.options = {
            _League: 'All',
            _LeagueId: '-1',
            _XmlPath: '',
            _XsltPath: ''
        };
        Object.extend(this.Options, options || {});
        this.GetFirst();
    },
    Cache: {},
    GetFrontPage: function(obj, leagueName) {
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        var divLoadingBull = $(this._Div_Load);
        divLoadingBull.style.visibility = "visible";
        divLoadingBull.style.display = "block";
        Object.extend(this.Options, {
            league: leagueName
        }
        || {});
        new Ajax.Request('/ajax/?a=[CMS]DBNewsbreaker_ChangeLeague', {
            method: 'post',
            parameters: this.Options,
            onSuccess: function(t) {
                var data = t.responseJSON;
                obj = $(obj);
                $(obj.parentNode).select('li').each(function(li) {
                    li.removeClassName('tabItemSelect');
                });
                obj.addClassName('tabItemSelect');
                $('fp_newsbreaker_data').update(data.html);
            }.bind(this),
            onComplete: function(t) {
                this.SendingAjax = false;
                this.GetFirst();
                divLoadingBull.style.visibility = "hidden";
                divLoadingBull.style.display = "none";
            }.bind(this)
        });
    },
    UpdateLeague: function(obj, leagueName) {
        this.GetFrontPage(obj, leagueName);
    },
    GetFirst: function() {
        this.DisplayNewsDescription((this.Options["name"] + "_DivDescription11"), (this.Options["name"] + "_DivCentralDescription"), $($((this.Options["name"] + "_DivDescription11")).parentNode).down('a'));
    },
    DisplayNewsDescription: function(pDivDescriptionId, pDivCentralDescriptionId, _obj) {
        var doc = document;
        var divDescriptionCtl = doc.getElementById(pDivDescriptionId);
        var divCentralDescriptionCtl = doc.getElementById(pDivCentralDescriptionId);
        if (divDescriptionCtl && divCentralDescriptionCtl)
            divCentralDescriptionCtl.innerHTML = divDescriptionCtl.innerHTML;
        if (Prototype != null && _obj != null) {
            $(_obj.parentNode.parentNode.parentNode).select('ul').each(function(ul, i) {
                ul.select('li').each(function(li)
                {
                    if (li.down('a') == null)
                        return;
                    li.down('a').removeClassName('newsbreaker_selected');
                });
            });
            $(_obj).addClassName('newsbreaker_selected');
        }
        return false;
    }
});
var DropDown = Class.create({
    initialize: function(options) {
        this.Options = options;
        this.showing = false;
        this.Timeout = 0;
        this.Effect = options.effects;
        this.AlignLeft = options.align == null || options.align == 'left' ? true : false;
        this.AlignCenter = options.align == 'center';
        this.Padding = options.padding == null ? {
            top: 0,
            left: 0,
            right: 0
        } : options.padding;
        this.Shadow = options.shadow == null ? null : options.shadow;
        this.CloseOnClick = options.closeOnClick != null && options.closeOnClick;
        this.CloseOnMouseout = options.closeOnMouseout != null && options.closeOnMouseout;
        this.ShowOnMouseover = options.showOnMouseover != null && options.showOnMouseover;
        this.Trigger = $(options.trigger);
        this.BaseElement = options.baseElement == 'parent' ? this.Trigger.parentNode : $(options.baseElement);
        this.DisplayElement = $(options.displayElement);
        this.hideListener = this.Hide.bindAsEventListener(this);
        this.showFinishListener = this.ShowFinish.bindAsEventListener(this);
        this.hideFinishListener = this.HideFinish.bindAsEventListener(this);
        this.HideDropDown = function() {
            clearTimeout(this.Timeout);
            this.DisplayElement.setStyle({
                display: 'none'
            });
            if (this.Shadow && this.Shadow.obj) {
                this.Shadow.obj.setStyle({
                    display: 'none'
                });
            }
        }.bind(this);
        this.MakeTrigger();
    },
    MakeTrigger: function() {
        if (this.Trigger == null)
            return;
        Event.observe(this.Trigger, 'click', this.TriggerClick.bind(this));
        if (this.ShowOnMouseover) {
            Event.observe(this.Trigger, 'mouseover', this.Show.bind(this));
            this.CloseOnMouseout = true;
        }
        var g = this;
        if (this.CloseOnMouseout) {
            Event.observe(this.DisplayElement, 'mouseout', function() {
                g.Timeout = setTimeout(g.Hide.bind(g), 300);
            });
            Event.observe(this.BaseElement, 'mouseout', function() {
                g.Timeout = setTimeout(g.Hide.bind(g), 300);
            });
            Event.observe(this.DisplayElement, 'mouseover', function() {
                clearTimeout(g.Timeout);
            });
            Event.observe(this.BaseElement, 'mouseover', function() {
                clearTimeout(g.Timeout);
            });
            if (this.ShowOnMouseover) {
                Event.observe(this.Trigger, 'mouseout', function() {
                    g.Timeout = setTimeout(g.Hide.bind(g), 300);
                });
                Event.observe(this.Trigger, 'mouseover', function() {
                    clearTimeout(g.Timeout);
                });
            }
        }
        if (!this.CloseOnClick)
            Event.observe(this.DisplayElement, 'click', this.Show.bind(this));
        Event.observe(this.BaseElement, 'click', function() {
            this.showing = true;
        });
        Event.observe(window, 'load', this.MoveDisplayElement.bind(this));
    },
    TriggerClick: function() {
        if (this.DisplayElement.getStyle('display') == 'block')
            this.Hide();
        else
            this.Show();
    },
    MoveDisplayElement: function() {
        $(document.body).appendChild(this.DisplayElement);
        if (this.Shadow != null) {
            var shadow = new Element('div').update('&nbsp;');
            shadow.setStyle({
                position: 'absolute',
                opacity: this.Shadow.opacity,
                background: '#000000',
                display: 'none'
            });
            this.Shadow.obj = shadow;
            $(document.body).appendChild(shadow);
        }
    },
    Show: function() {
        Event.observe(document.body, 'click', this.hideListener);
        this.ShowBegin();
        var pos = this.BaseElement.cumulativeOffset();
        var hasEffect = this.Effect != null && this.Effect.show != 'none' && this.Effect.show != null && this.DisplayElement.getStyle('display') == 'none';
        var anchorRight = pos.left + this.DisplayElement.getWidth() > document.viewport.getWidth();
        this.DisplayElement.setStyle({
            position: 'absolute',
            top: pos.top + this.BaseElement.getHeight() + this.Padding.top + 'px',
            display: 'none',
            zIndex: 1000
        });
        if (this.AlignLeft) {
            this.DisplayElement.setStyle({
                left: anchorRight ? pos.left + this.BaseElement.getWidth() - this.DisplayElement.getWidth() + this.Padding.left + 'px': pos.left + this.Padding.left + 'px'
            });
        } else if (this.AlignCenter) {
            this.DisplayElement.setStyle({
                left: pos.left + this.BaseElement.getWidth() / 2 - this.DisplayElement.getWidth() / 2 + this.Padding.left + 'px'
            });
        } else {
            this.DisplayElement.setStyle({
                right: anchorRight ? document.viewport.getWidth() - (pos.left + this.BaseElement.getWidth()) + this.Padding.right + 'px': document.viewport.getWidth() - (pos.left + this.DisplayElement.getWidth()) + this.Padding.right + 'px'
            });
        }
        this.showing = true;
        if (hasEffect)
            Effect.toggle(this.DisplayElement, this.Effect.show, {
                duration: this.Effect.duration,
                afterFinish: this.showFinishListener
            });
        else {
            this.DisplayElement.setStyle({
                display: 'block'
            });
            this.ShowFinish();
        }
    },
    ResizeShadow: function() {
        this.ShowShadow();
    },
    ShowShadow: function() {
        var pos = this.DisplayElement.cumulativeOffset();
        if (pos.top == 0 && pos.left == 0)
            return;
        this.Shadow.obj.setStyle({
            top: pos.top + this.Shadow.offset + 'px',
            left: pos.left + this.Shadow.offset + 'px',
            width: this.DisplayElement.getWidth() + 'px',
            height: this.DisplayElement.getHeight() + 'px',
            display: 'block'
        });
    },
    ShowBegin: function() {
        if (this.Options.onShowBegin)
            this.Options.onShowBegin(this);
    },
    ShowFinish: function() {
        if (this.Shadow != null)
            this.ShowShadow();
        if (this.Options.onShowFinish)
            this.Options.onShowFinish(this);
    },
    Close: function() {
        this.Hide();
    },
    Hide: function() {
        if (this.showing) {
            this.showing = false;
            return;
        }
        if (this.Shadow != null && this.Shadow.obj != null)
            this.Shadow.obj.setStyle({
                display: 'none'
            });
        var hasEffect = this.Effect != null && this.Effect.hide != 'none' && this.Effect.hide != null && this.DisplayElement.getStyle('display') == 'block';
        if (hasEffect)
            Effect.toggle(this.DisplayElement, this.Effect.hide, {
                duration: this.Effect.duration,
                afterFinish: this.hideFinishListener
            });
        else {
            this.DisplayElement.setStyle({
                display: !hasEffect ? 'none': 'block'
            });
            this.HideFinish();
        }
        Event.stopObserving(document.body, 'click', this.hideListener);
    },
    HideFinish: function() {
        if (this.Options.onHideFinish)
            this.Options.onHideFinish(this);
    }
});
var UserDropDown = Class.create(DropDown, {
    initialize: function($super, userId) {
        $super({
            trigger: 'userDropDown_Trigger_' + userId,
            baseElement: 'parent',
            displayElement: 'userDropDown_' + userId,
            closeOnClick: false,
            align: 'right',
            padding: {
                top: 1,
                left: 0,
                right: 0
            },
            shadow: {
                opacity: .2,
                offset: 2
            },
            effects: {
                show: 'blind',
                hide: 'none',
                duration: .2
            },
            onShowBegin: function(t) {
                t.Trigger.addClassName('userDropDown_Over');
            },
            onShowFinish: function(t) {},
            onHideFinish: function(t) {
                t.Trigger.removeClassName('userDropDown_Over');
            }
        });
    }
});
(function() {
    function g(o) {
        console.log("$f.fireEvent", [].slice.call(o))
    }
    function k(q) {
        if (!q || typeof q != "object") {
            return q
        }
        var o = new q.constructor();
        for (var p in q) {
            if (q.hasOwnProperty(p)) {
                o[p] = k(q[p])
            }
        }
        return o
    }
    function m(t, q) {
        if (!t) {
            return 
        }
        var o, p = 0, r = t.length;
        if (r === undefined) {
            for (o in t) {
                if (q.call(t[o], o, t[o]) === false) {
                    break
                }
            }
        } else {
            for (var s = t[0]; p < r && q.call(s, p, s) !== false; s = t[++p]) {}
        }
        return t
    }
    function c(o) {
        return document.getElementById(o)
    }
    function i(q, p, o) {
        if (typeof p != "object") {
            return q
        }
        if (q && p) {
            m(p, function(r, s) {
                if (!o || typeof s != "function") {
                    q[r] = s
                }
            })
        }
        return q
    }
    function n(s) {
        var q = s.indexOf(".");
        if (q!=-1) {
            var p = s.substring(0, q) || "*";
            var o = s.substring(q + 1, s.length);
            var r = [];
            m(document.getElementsByTagName(p), function() {
                if (this.className && this.className.indexOf(o)!=-1) {
                    r.push(this)
                }
            });
            return r
        }
    }
    function f(o) {
        o = o || window.event;
        if (o.preventDefault) {
            o.stopPropagation();
            o.preventDefault()
        } else {
            o.returnValue = false;
            o.cancelBubble = true
        }
        return false
    }
    function j(q, o, p) {
        q[o] = q[o] || [];
        q[o].push(p)
    }
    function e() {
        return "_" + ("" + Math.random()).substring(2, 10)
    }
    var h = function(t, r, s) {
        var q = this;
        var p = {};
        var u = {};
        q.index = r;
        if (typeof t == "string") {
            t = {
                url: t
            }
        }
        i(this, t, true);
        m(("Begin*,Start,Pause*,Resume*,Seek*,Stop*,Finish*,LastSecond,Update,BufferFull,BufferEmpty,BufferStop").split(","), function() {
            var v = "on" + this;
            if (v.indexOf("*")!=-1) {
                v = v.substring(0, v.length - 1);
                var w = "onBefore" + v.substring(2);
                q[w] = function(x) {
                    j(u, w, x);
                    return q
                }
            }
            q[v] = function(x) {
                j(u, v, x);
                return q
            };
            if (r==-1) {
                if (q[w]) {
                    s[w] = q[w]
                }
                if (q[v]) {
                    s[v] = q[v]
                }
            }
        });
        i(this, {
            onCuepoint: function(x, w) {
                if (arguments.length == 1) {
                    p.embedded = [null, x];
                    return q
                }
                if (typeof x == "number") {
                    x = [x]
                }
                var v = e();
                p[v] = [x, w];
                if (s.isLoaded()) {
                    s._api().fp_addCuepoints(x, r, v)
                }
                return q
            },
            update: function(w) {
                i(q, w);
                if (s.isLoaded()) {
                    s._api().fp_updateClip(w, r)
                }
                var v = s.getConfig();
                var x = (r==-1) ? v.clip: v.playlist[r];
                i(x, w, true)
            },
            _fireEvent: function(v, y, w, A) {
                if (v == "onLoad") {
                    m(p, function(B, C) {
                        if (C[0]) {
                            s._api().fp_addCuepoints(C[0], r, B)
                        }
                    });
                    return false
                }
                A = A || q;
                if (v == "onCuepoint") {
                    var z = p[y];
                    if (z) {
                        return z[1].call(s, A, w)
                    }
                }
                if (y && "onBeforeBegin,onMetaData,onStart,onUpdate,onResume".indexOf(v)!=-1) {
                    i(A, y);
                    if (y.metaData) {
                        if (!A.duration) {
                            A.duration = y.metaData.duration
                        } else {
                            A.fullDuration = y.metaData.duration
                        }
                    }
                }
                var x = true;
                m(u[v], function() {
                    x = this.call(s, A, y, w)
                });
                return x
            }
        });
        if (t.onCuepoint) {
            var o = t.onCuepoint;
            q.onCuepoint.apply(q, typeof o == "function" ? [o] : o);
            delete t.onCuepoint
        }
        m(t, function(v, w) {
            if (typeof w == "function") {
                j(u, v, w);
                delete t[v]
            }
        });
        if (r==-1) {
            s.onCuepoint = this.onCuepoint
        }
    };
    var l = function(p, r, q, t) {
        var s = {};
        var o = this;
        var u = false;
        if (t) {
            i(s, t)
        }
        m(r, function(v, w) {
            if (typeof w == "function") {
                s[v] = w;
                delete r[v]
            }
        });
        i(this, {
            animate: function(y, z, x) {
                if (!y) {
                    return o
                }
                if (typeof z == "function") {
                    x = z;
                    z = 500
                }
                if (typeof y == "string") {
                    var w = y;
                    y = {};
                    y[w] = z;
                    z = 500
                }
                if (x) {
                    var v = e();
                    s[v] = x
                }
                if (z === undefined) {
                    z = 500
                }
                r = q._api().fp_animate(p, y, z, v);
                return o
            },
            css: function(w, x) {
                if (x !== undefined) {
                    var v = {};
                    v[w] = x;
                    w = v
                }
                r = q._api().fp_css(p, w);
                i(o, r);
                return o
            },
            show: function() {
                this.display = "block";
                q._api().fp_showPlugin(p);
                return o
            },
            hide: function() {
                this.display = "none";
                q._api().fp_hidePlugin(p);
                return o
            },
            toggle: function() {
                this.display = q._api().fp_togglePlugin(p);
                return o
            },
            fadeTo: function(y, x, w) {
                if (typeof x == "function") {
                    w = x;
                    x = 500
                }
                if (w) {
                    var v = e();
                    s[v] = w
                }
                this.display = q._api().fp_fadeTo(p, y, x, v);
                this.opacity = y;
                return o
            },
            fadeIn: function(w, v) {
                return o.fadeTo(1, w, v)
            },
            fadeOut: function(w, v) {
                return o.fadeTo(0, w, v)
            },
            getName: function() {
                return p
            },
            getPlayer: function() {
                return q
            },
            _fireEvent: function(w, v, x) {
                if (w == "onUpdate") {
                    var y = q._api().fp_getPlugin(p);
                    if (!y) {
                        return 
                    }
                    i(o, y);
                    delete o.methods;
                    if (!u) {
                        m(y.methods, function() {
                            var A = "" + this;
                            o[A] = function() {
                                var B = [].slice.call(arguments);
                                var C = q._api().fp_invoke(p, A, B);
                                return C === "undefined" || C === undefined ? o : C
                            }
                        });
                        u = true
                    }
                }
                var z = s[w];
                if (z) {
                    z.apply(o, v);
                    if (w.substring(0, 1) == "_") {
                        delete s[w]
                    }
                }
            }
        })
    };
    function b(o, t, z) {
        var E = this, y = null, x, u, p = [], s = {}, B = {}, r, v, w, D, A, q;
        i(E, {
            id: function() {
                return r
            },
            isLoaded: function() {
                return (y !== null)
            },
            getParent: function() {
                return o
            },
            hide: function(F) {
                if (F) {
                    o.style.height = "0px"
                }
                if (y) {
                    y.style.height = "0px"
                }
                return E
            },
            show: function() {
                o.style.height = q + "px";
                if (y) {
                    y.style.height = A + "px"
                }
                return E
            },
            isHidden: function() {
                return y && parseInt(y.style.height, 10) === 0
            },
            load: function(F) {
                if (!y && E._fireEvent("onBeforeLoad") !== false) {
                    m(a, function() {
                        this.unload()
                    });
                    x = o.innerHTML;
                    if (x&&!flashembed.isSupported(t.version)) {
                        o.innerHTML = ""
                    }
                    flashembed(o, t, {
                        config: z
                    });
                    if (F) {
                        F.cached = true;
                        j(B, "onLoad", F)
                    }
                }
                return E
            },
            unload: function() {
                if (x.replace(/\s/g, "") !== "") {
                    if (E._fireEvent("onBeforeUnload") === false) {
                        return E
                    }
                    try {
                        if (y) {
                            y.fp_close();
                            E._fireEvent("onUnload")
                        }
                    } catch (F) {}
                    y = null;
                    o.innerHTML = x
                }
                return E
            },
            getClip: function(F) {
                if (F === undefined) {
                    F = D
                }
                return p[F]
            },
            getCommonClip: function() {
                return u
            },
            getPlaylist: function() {
                return p
            },
            getPlugin: function(F) {
                var H = s[F];
                if (!H && E.isLoaded()) {
                    var G = E._api().fp_getPlugin(F);
                    if (G) {
                        H = new l(F, G, E);
                        s[F] = H
                    }
                }
                return H
            },
            getScreen: function() {
                return E.getPlugin("screen")
            },
            getControls: function() {
                return E.getPlugin("controls")
            },
            getConfig: function(F) {
                return F ? k(z) : z
            },
            getFlashParams: function() {
                return t
            },
            loadPlugin: function(I, H, K, J) {
                if (typeof K == "function") {
                    J = K;
                    K = {}
                }
                var G = J ? e(): "_";
                E._api().fp_loadPlugin(I, H, K, G);
                var F = {};
                F[G] = J;
                var L = new l(I, null, E, F);
                s[I] = L;
                return L
            },
            getState: function() {
                return y ? y.fp_getState() : - 1
            },
            play: function(G, F) {
                function H() {
                    if (G !== undefined) {
                        E._api().fp_play(G, F)
                    } else {
                        E._api().fp_play()
                    }
                }
                if (y) {
                    H()
                } else {
                    E.load(function() {
                        H()
                    })
                }
                return E
            },
            getVersion: function() {
                var G = "flowplayer.js 3.1.4";
                if (y) {
                    var F = y.fp_getVersion();
                    F.push(G);
                    return F
                }
                return G
            },
            _api: function() {
                if (!y) {
                    throw "Flowplayer " + E.id() + " not loaded when calling an API method"
                }
                return y
            },
            setClip: function(F) {
                E.setPlaylist([F]);
                return E
            },
            getIndex: function() {
                return w
            }
        });
        m(("Click*,Load*,Unload*,Keypress*,Volume*,Mute*,Unmute*,PlaylistReplace,ClipAdd,Fullscreen*,FullscreenExit,Error,MouseOver,MouseOut").split(","), function() {
            var F = "on" + this;
            if (F.indexOf("*")!=-1) {
                F = F.substring(0, F.length - 1);
                var G = "onBefore" + F.substring(2);
                E[G] = function(H) {
                    j(B, G, H);
                    return E
                }
            }
            E[F] = function(H) {
                j(B, F, H);
                return E
            }
        });
        m(("pause,resume,mute,unmute,stop,toggle,seek,getStatus,getVolume,setVolume,getTime,isPaused,isPlaying,startBuffering,stopBuffering,isFullscreen,toggleFullscreen,reset,close,setPlaylist,addClip,playFeed").split(","), function() {
            var F = this;
            E[F] = function(H, G) {
                if (!y) {
                    return E
                }
                var I = null;
                if (H !== undefined && G !== undefined) {
                    I = y["fp_" + F](H, G)
                } else {
                    I = (H === undefined) ? y["fp_" + F]() : y["fp_" + F](H)
                }
                return I === "undefined" || I === undefined ? E : I
            }
        });
        E._fireEvent = function(O) {
            if (typeof O == "string") {
                O = [O]
            }
            var P = O[0], M = O[1], K = O[2], J = O[3], I = 0;
            if (z.debug) {
                g(O)
            }
            if (!y && P == "onLoad" && M == "player") {
                y = y || c(v);
                A = y.clientHeight;
                m(p, function() {
                    this._fireEvent("onLoad")
                });
                m(s, function(Q, R) {
                    R._fireEvent("onUpdate")
                });
                u._fireEvent("onLoad")
            }
            if (P == "onLoad" && M != "player") {
                return 
            }
            if (P == "onError") {
                if (typeof M == "string" || (typeof M == "number" && typeof K == "number")) {
                    M = K;
                    K = J
                }
            }
            if (P == "onContextMenu") {
                m(z.contextMenu[M], function(Q, R) {
                    R.call(E)
                });
                return 
            }
            if (P == "onPluginEvent") {
                var F = M.name || M;
                var G = s[F];
                if (G) {
                    G._fireEvent("onUpdate", M);
                    G._fireEvent(K, O.slice(3))
                }
                return 
            }
            if (P == "onPlaylistReplace") {
                p = [];
                var L = 0;
                m(M, function() {
                    p.push(new h(this, L++, E))
                })
            }
            if (P == "onClipAdd") {
                if (M.isInStream) {
                    return 
                }
                M = new h(M, K, E);
                p.splice(K, 0, M);
                for (I = K + 1; I < p.length; I++) {
                    p[I].index++
                }
            }
            var N = true;
            if (typeof M == "number" && M < p.length) {
                D = M;
                var H = p[M];
                if (H) {
                    N = H._fireEvent(P, K, J)
                }
                if (!H || N !== false) {
                    N = u._fireEvent(P, K, J, H)
                }
            }
            m(B[P], function() {
                N = this.call(E, M, K);
                if (this.cached) {
                    B[P].splice(I, 1)
                }
                if (N === false) {
                    return false
                }
                I++
            });
            return N
        };
        function C() {
            if ($f(o)) {
                $f(o).getParent().innerHTML = "";
                w = $f(o).getIndex();
                a[w] = E
            } else {
                a.push(E);
                w = a.length - 1
            }
            q = parseInt(o.style.height, 10) || o.clientHeight;
            if (typeof t == "string") {
                t = {
                    src: t
                }
            }
            r = o.id || "fp" + e();
            v = t.id || r + "_api";
            t.id = v;
            z.playerId = r;
            if (typeof z == "string") {
                z = {
                    clip: {
                        url: z
                    }
                }
            }
            if (typeof z.clip == "string") {
                z.clip = {
                    url: z.clip
                }
            }
            z.clip = z.clip || {};
            if (o.getAttribute("href", 2)&&!z.clip.url) {
                z.clip.url = o.getAttribute("href", 2)
            }
            u = new h(z.clip, - 1, E);
            z.playlist = z.playlist || [z.clip];
            var F = 0;
            m(z.playlist, function() {
                var H = this;
                if (typeof H == "object" && H.length) {
                    H = {
                        url: "" + H
                    }
                }
                m(z.clip, function(I, J) {
                    if (J !== undefined && H[I] === undefined && typeof J != "function") {
                        H[I] = J
                    }
                });
                z.playlist[F] = H;
                H = new h(H, F, E);
                p.push(H);
                F++
            });
            m(z, function(H, I) {
                if (typeof I == "function") {
                    if (u[H]) {
                        u[H](I)
                    } else {
                        j(B, H, I)
                    }
                    delete z[H]
                }
            });
            m(z.plugins, function(H, I) {
                if (I) {
                    s[H] = new l(H, I, E)
                }
            });
            if (!z.plugins || z.plugins.controls === undefined) {
                s.controls = new l("controls", null, E)
            }
            s.canvas = new l("canvas", null, E);
            t.bgcolor = t.bgcolor || "#000000";
            t.version = t.version || [9, 0];
            t.expressInstall = "http://www.flowplayer.org/swf/expressinstall.swf";
            function G(H) {
                if (!E.isLoaded() && E._fireEvent("onBeforeClick") !== false) {
                    E.load()
                }
                return f(H)
            }
            x = o.innerHTML;
            if (x.replace(/\s/g, "") !== "") {
                if (o.addEventListener) {
                    o.addEventListener("click", G, false)
                } else {
                    if (o.attachEvent) {
                        o.attachEvent("onclick", G)
                    }
                }
            } else {
                if (o.addEventListener) {
                    o.addEventListener("click", f, false)
                }
                E.load()
            }
        }
        if (typeof o == "string") {
            flashembed.domReady(function() {
                var F = c(o);
                if (!F) {
                    throw "Flowplayer cannot access element: " + o
                } else {
                    o = F;
                    C()
                }
            })
        } else {
            C()
        }
    }
    var a = [];
    function d(o) {
        this.length = o.length;
        this.each = function(p) {
            m(o, p)
        };
        this.size = function() {
            return o.length
        }
    }
    window.flowplayer = window.$f = function() {
        var p = null;
        var o = arguments[0];
        if (!arguments.length) {
            m(a, function() {
                if (this.isLoaded()) {
                    p = this;
                    return false
                }
            });
            return p || a[0]
        }
        if (arguments.length == 1) {
            if (typeof o == "number") {
                return a[o]
            } else {
                if (o == "*") {
                    return new d(a)
                }
                m(a, function() {
                    if (this.id() == o.id || this.id() == o || this.getParent() == o) {
                        p = this;
                        return false
                    }
                });
                return p
            }
        }
        if (arguments.length > 1) {
            var r = arguments[1];
            var q = (arguments.length == 3) ? arguments[2]: {};
            if (typeof o == "string") {
                if (o.indexOf(".")!=-1) {
                    var t = [];
                    m(n(o), function() {
                        t.push(new b(this, k(r), k(q)))
                    });
                    return new d(t)
                } else {
                    var s = c(o);
                    return new b(s !== null ? s : o, r, q)
                }
            } else {
                if (o) {
                    return new b(o, r, q)
                }
            }
        }
        return null
    };
    i(window.$f, {
        fireEvent: function() {
            var o = [].slice.call(arguments);
            var q = $f(o[0]);
            return q ? q._fireEvent(o.slice(1)) : null
        },
        addPlugin: function(o, p) {
            b.prototype[o] = p;
            return $f
        },
        each: m,
        extend: i
    });
    if (typeof jQuery == "function") {
        jQuery.prototype.flowplayer = function(q, p) {
            if (!arguments.length || typeof arguments[0] == "number") {
                var o = [];
                this.each(function() {
                    var r = $f(this);
                    if (r) {
                        o.push(r)
                    }
                });
                return arguments.length ? o[arguments[0]] : new d(o)
            }
            return this.each(function() {
                $f(this, k(q), p ? k(p) : {})
            })
        }
    }
})();
(function() {
    var e = typeof jQuery == "function";
    var i = {
        width: "100%",
        height: "100%",
        allowfullscreen: true,
        allowscriptaccess: "always",
        quality: "high",
        version: null,
        onFail: null,
        expressInstall: null,
        w3c: false,
        cachebusting: false
    };
    if (e) {
        jQuery.tools = jQuery.tools || {};
        jQuery.tools.flashembed = {
            version: "1.0.4",
            conf: i
        }
    }
    function j() {
        if (c.done) {
            return false
        }
        var l = document;
        if (l && l.getElementsByTagName && l.getElementById && l.body) {
            clearInterval(c.timer);
            c.timer = null;
            for (var k = 0; k < c.ready.length; k++) {
                c.ready[k].call()
            }
            c.ready = null;
            c.done = true
        }
    }
    var c = e ? jQuery: function(k) {
        if (c.done) {
            return k()
        }
        if (c.timer) {
            c.ready.push(k)
        } else {
            c.ready = [k];
            c.timer = setInterval(j, 13)
        }
    };
    function f(l, k) {
        if (k) {
            for (key in k) {
                if (k.hasOwnProperty(key)) {
                    l[key] = k[key]
                }
            }
        }
        return l
    }
    function g(k) {
        switch (h(k)) {
        case"string":
            k = k.replace(new RegExp('(["\\\\])', "g"), "\\$1");
            k = k.replace(/^\s?(\d+)%/, "$1pct");
            return '"' + k + '"';
        case"array":
            return "[" + b(k, function(n) {
                return g(n)
            }).join(",") + "]";
        case"function":
            return '"function()"';
        case"object":
            var l = [];
            for (var m in k) {
                if (k.hasOwnProperty(m)) {
                    l.push('"' + m + '":' + g(k[m]))
                }
            }
            return "{" + l.join(",") + "}"
        }
        return String(k).replace(/\s/g, " ").replace(/\'/g, '"')
    }
    function h(l) {
        if (l === null || l === undefined) {
            return false
        }
        var k = typeof l;
        return (k == "object" && l.push) ? "array" : k
    }
    if (window.attachEvent) {
        window.attachEvent("onbeforeunload", function() {
            __flash_unloadHandler = function() {};
            __flash_savedUnloadHandler = function() {}
        })
    }
    function b(k, n) {
        var m = [];
        for (var l in k) {
            if (k.hasOwnProperty(l)) {
                m[l] = n(k[l])
            }
        }
        return m
    }
    function a(r, t) {
        var q = f({}, r);
        var s = document.all;
        var n = '<object width="' + q.width + '" height="' + q.height + '"';
        if (s&&!q.id) {
            q.id = "_" + ("" + Math.random()).substring(9)
        }
        if (q.id) {
            n += ' id="' + q.id + '"'
        }
        if (q.cachebusting) {
            q.src += ((q.src.indexOf("?")!=-1 ? "&" : "?") + Math.random())
        }
        if (q.w3c ||!s) {
            n += ' data="' + q.src + '" type="application/x-shockwave-flash"'
        } else {
            n += ' classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000"'
        }
        n += ">";
        if (q.w3c || s) {
            n += '<param name="movie" value="' + q.src + '" />'
        }
        q.width = q.height = q.id = q.w3c = q.src = null;
        for (var l in q) {
            if (q[l] !== null) {
                n += '<param name="' + l + '" value="' + q[l] + '" />'
            }
        }
        var o = "";
        if (t) {
            for (var m in t) {
                if (t[m] !== null) {
                    o += m + "=" + (typeof t[m] == "object" ? g(t[m]) : t[m]) + "&"
                }
            }
            o = o.substring(0, o.length - 1);
            n += '<param name="flashvars" value=\'' + o + "' />"
        }
        n += "</object>";
        return n
    }
    function d(m, p, l) {
        var k = flashembed.getVersion();
        f(this, {
            getContainer: function() {
                return m
            },
            getConf: function() {
                return p
            },
            getVersion: function() {
                return k
            },
            getFlashvars: function() {
                return l
            },
            getApi: function() {
                return m.firstChild
            },
            getHTML: function() {
                return a(p, l)
            }
        });
        var q = p.version;
        var r = p.expressInstall;
        var o=!q || flashembed.isSupported(q);
        if (o) {
            p.onFail = p.version = p.expressInstall = null;
            m.innerHTML = a(p, l)
        } else {
            if (q && r && flashembed.isSupported([6, 65])) {
                f(p, {
                    src: r
                });
                l = {
                    MMredirectURL: location.href,
                    MMplayerType: "PlugIn",
                    MMdoctitle: document.title
                };
                m.innerHTML = a(p, l)
            } else {
                if (m.innerHTML.replace(/\s/g, "") !== "") {} else {
                    m.innerHTML = "<h2>Flash version " + q + " or greater is required</h2><h3>" + (k[0] > 0 ? "Your version is " + k : "You have no flash plugin installed") + "</h3>" + (m.tagName == "A" ? "<p>Click here to download latest version</p>" : "<p>Download latest version from <a href='http://www.adobe.com/go/getflashplayer'>here</a></p>");
                    if (m.tagName == "A") {
                        m.onclick = function() {
                            location.href = "http://www.adobe.com/go/getflashplayer"
                        }
                    }
                }
            }
        }
        if (!o && p.onFail) {
            var n = p.onFail.call(this);
            if (typeof n == "string") {
                m.innerHTML = n
            }
        }
        if (document.all) {
            window[p.id] = document.getElementById(p.id)
        }
    }
    window.flashembed = function(l, m, k) {
        if (typeof l == "string") {
            var n = document.getElementById(l);
            if (n) {
                l = n
            } else {
                c(function() {
                    flashembed(l, m, k)
                });
                return 
            }
        }
        if (!l) {
            return 
        }
        if (typeof m == "string") {
            m = {
                src: m
            }
        }
        var o = f({}, i);
        f(o, m);
        return new d(l, o, k)
    };
    f(window.flashembed, {
        getVersion: function() {
            var m = [0, 0];
            if (navigator.plugins && typeof navigator.plugins["Shockwave Flash"] == "object") {
                var l = navigator.plugins["Shockwave Flash"].description;
                if (typeof l != "undefined") {
                    l = l.replace(/^.*\s+(\S+\s+\S+$)/, "$1");
                    var n = parseInt(l.replace(/^(.*)\..*$/, "$1"), 10);
                    var r = /r/.test(l) ? parseInt(l.replace(/^.*r(.*)$/, "$1"), 10): 0;
                    m = [n, r]
                }
            } else {
                if (window.ActiveXObject) {
                    try {
                        var p = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7")
                    } catch (q) {
                        try {
                            p = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6");
                            m = [6, 0];
                            p.AllowScriptAccess = "always"
                        } catch (k) {
                            if (m[0] == 6) {
                                return m
                            }
                        }
                        try {
                            p = new ActiveXObject("ShockwaveFlash.ShockwaveFlash")
                        } catch (o) {}
                    }
                    if (typeof p == "object") {
                        l = p.GetVariable("$version");
                        if (typeof l != "undefined") {
                            l = l.replace(/^\S+\s+(.*)$/, "$1").split(",");
                            m = [parseInt(l[0], 10), parseInt(l[2], 10)]
                        }
                    }
                }
            }
            return m
        },
        isSupported: function(k) {
            var m = flashembed.getVersion();
            var l = (m[0] > k[0]) || (m[0] == k[0] && m[1] >= k[1]);
            return l
        },
        domReady: c,
        asString: g,
        getHTML: a
    });
    if (e) {
        jQuery.fn.flashembed = function(l, k) {
            var m = null;
            this.each(function() {
                m = flashembed(this, l, k)
            });
            return l.api === false ? this : m
        }
    }
})();
eval(function(p, a, c, k, e, d) {
    e = function(c) {
        return (c < a ? '' : e(parseInt(c / a))) + ((c = c%a) > 35 ? String.fromCharCode(c + 29) : c.toString(36))
    };
    while (c--)
        if (k[c])
            p = p.replace(new RegExp('\\b' + e(c) + '\\b', 'g'), k[c]);
    return p
}('u={};u.49=\'0.2\';5(!13.1p)13.1p=4(o){B.y(13.48,o)};16={47:4(3){3=$(3);19(3=3.46)5(3.2m==1)d $(3);d 1a},45:4(3){3=$(3);19(3=3.2l)5(3.2m==1)d $(3);d 1a},44:4(3){d $(3).2k.43(3)},42:4(3,17){d $(3).2l.41(17)},40:4(3,17){$(3).2k.3Z(17,3);d 17}};13.1p(16);16.1F={2j:{\'3Y\':\'3X\',\'H\':\'3W\'},2i:4(h,j,7){8 1I;5(1I=6.2j[j])7[1I]=h[j];l 5(j==\'1G\')7.1G.3V=h[j];l 5(j.1z(/^Z/))7[j]=1r 3U(h[j]);l 7.2h(j,h[j])},2e:4(t){d 4(){8 h,n;5(G.w>0){5(G[0].3T||1d G[0]=="2g")n=G;l{h=G[0];n=[].3S.1B(G,1)}}d 16.1F.1b(t,h,n)}},1b:4(t,h,n){h=h||{};n=n||[];8 1H=2a.29.1z(/3R/);8 7=S.3Q((1H&&h.1n)?"<"+t+" 1n="+h.1n+">":t);H(8 j 1e h){5(1d h[j]!=\'4\'){5(1H)6.2i(h,h,7);l 7.2h(j,h[j])}}H(8 i=0;i<n.w;i++){5(1d n[i]==\'2g\')n[i]=S.3P(n[i]);7.3O(n[i])}d $(7)}};(4(){8 2f=("p|3N|3M|3L|3K|3J|3I|3H|3G|3F|3E|3D|3C|3B|3A|"+"3z|3y|3x|3w|3v|3u|3t|3s|3r|3q|3p|3o|3n|3m|"+"1u|3l|3k|3j|3i|3h|3g|3f|3e|3d|a|3c|3b|3a|"+"39|38|1G|37|36|35|34|33|32|31|30|2Z|"+"2Y|2X|2W|2V|8").18("|");8 7,i=0;19(7=2f[i++])1g[\'$\'+7]=16.1F.2e(7)})();B.y(9,{2c:4(){5(G.2d.1E)d;G.2d.1E=N;5(9.1D)2U(9.1D);9.14.P(4(f){f()});9.14=1a},21:4(f){5(!6.14){8 15=6.2c;5(15.1E)d f();5(S.2b)S.2b("2T",15,M);5(/2S/i.28(2a.29)){6.1D=2R(4(){5(/2Q|2P/.28(S.2O))15()},10)}9.I(1g,\'1i\',15);9.14=[]}9.14.Y(f)}});13.1p({I:4(7,m,1o){9.I(7,m,1o)},1h:4(7,m,1o){9.1h(7,m,1o)}});B.y(9,{I:4(7,r,F){7=$(7);5(!F.$$1m)F.$$1m=9.22++;5(!7.E)7.E={};8 D=7.E[r];5(!D){D=7.E[r]={};5(7["Z"+r]){D[0]=7["Z"+r]}}D[F.$$1m]=F;7["Z"+r]=9.27;5(!9.1k)9.1k=[];9.1k.Y([7,1n,F,M])},1h:4(7,r,F){5(7.E&&7.E[r])2N 7.E[r][F.$$1m]},27:4(e){8 1l=N;e=e||9.25(1g.m);8 D=6.E[e.r],7=$(6);H(8 i 1e D){7.$$26=D[i];5(7.$$26(e)===M)1l=M}d 1l},25:4(e){e.2M=9.24;e.2L=9.23;d e},24:4(){6.1l=M},23:4(){6.2K=N},22:1});B.y(9,{2J:4(3,m,1j){3=$(3);1j=1j||{r:m};6.1k.P(4(L){5(L[0]==3&&L[1]==m)L[2].1B(3,1j)})}});9.J=4(U){8 V=6.J;B.y(V.U,U);5(V.1Z){6.21(V.1i.1s(V))}2I.2H.2G({2F:4(){5(9.J.20)2E(4(){V.1i()},10)}})};B.y(9.J,{U:{},L:[],20:N,1Z:N,1i:4(){6.1f();H(8 g 1e 6.U){8 C=6.U[g];8 1Y=g.18(\',\');1Y.P(4(1X){8 1C=1X.18(/:(?=[a-z]+$)/),1W=1C[0],m=1C[1];$$(1W).P(4(3){5(m){$(3).I(m,C);9.J.L.Y([3,m,C])}l{5(!3.$$12||!3.$$12.1N(C)){5(C.1A)C.1A(3);l C.1B($(3));3.$$12=3.$$12||[];3.$$12.Y(C)}}})})}},1f:4(){6.L.P(4(c){9.1h.1S(9,c)})}});9.I(1g,\'1f\',9.J.1f.1s(9.J));1U={1b:4(1V){8 11=4(3){6.3=$(3)};11.1y.1x=2D.K;B.y(11.1y,1V);B.y(11,1U.1T);d 11},1T:{1A:4(3){8 q=1r 6(3);q.1x.1S(q);6.1R(q);d q},1R:4(q){H(8 1c 1e q)5(1c.1z(/^Z(.+)/)&&1d q[1c]==\'4\')q.3.I(2C.$1,q[1c].2B(q))}}};u.1q=2A.1b();u.1q.1y={1x:4(x){6.k=[];6.x=[];6.W=0;H(8 i=x.w-1;i>=0;i--){8 T={t:\'*\',v:1a,R:[]};8 g=x[i];8 2z=g.w-1;2y{8 v=g.1Q("#");8 Q=g.1Q(".");8 X=2x.2w(v,Q);5(X==-1)T.t=g.2v();l 5(v==-1||Q==X)T.R.Y(g.1w(Q+1));l 5(!T.v)T.v=g.1w(v+1);g=g.1w(0,X)}19(X>0);6.x[i]=T}},1J:4(1P){6.1t(1P||S,6.W==(6.x.w-1));d 6.k},1t:4(1v,1M){8 g=6.x[6.W],k=[],3;5(g.v){3=$(g.v);5(3&&(g.t==\'*\'||3.2u==g.t)&&(3.2t(1v))){k=[3]}}l{k=$A(1v.2s(g.t))}5(g.R.w==1){k=k.1u(4(O){d $(O).2r(g.R[0])})}l 5(g.R.w>1){k=k.1u(4(O){8 1O=$(O).2q();d g.R.2p(4(Q){d 1O.1N(Q)})})}5(1M){6.k=6.k.2o(k)}l{++6.W;k.P(4(O){6.1t(O,6.W==(6.x.w-1))}.1s(6))}}};u.$$1K=$$;u.1L$$=N;$$=4(a,b){5(u.1L$$==M||b||a.2n("[")>=0)d u.$$1K(a,b);d 1r u.1q(a.18(/\\s+/)).1J()};', 62, 258, '|||element|function|if|this|el|var|Event||||return|||selector|attrs||attr|results|else|event|children|||bound|type||tag|LowPro|id|length|selectors|extend|||Object|observer|handlers|events|func|arguments|for|observe|addBehavior||cache|false|true|target|each|klass|classes|document|params|rules|ab|index|cursor|push|on||behavior|assigned|Element|_readyCallbacks|domReady|DOM|node|split|while|null|create|member|typeof|in|unload|window|stopObserving|load|fakeEvent|observers|returnValue|guid|name|callback|addMethods|SelectorLite|new|bind|findElements|select|parent|substring|initialize|prototype|match|attach|call|parts|_timer|done|Builder|style|isIE|trans|get|old|optimize|descendant|include|klasses|root|lastIndexOf|_bindEvents|apply|ClassMethods|Behavior|members|css|sel|sels|autoTrigger|reassignAfterAjax|onReady|_guid|_stopPropagation|_preventDefault|_fixEvent|handleEvent|_handleEvent|test|userAgent|navigator|addEventListener|_domReady|callee|tagFunc|els|string|setAttribute|ieAttrSet|IE_TRANSLATIONS|parentNode|previousSibling|nodeType|indexOf|concat|all|classNames|hasClassName|getElementsByTagName|childOf|tagName|toUpperCase|max|Math|do|needle|Class|bindAsEventListener|RegExp|Prototype|setTimeout|onComplete|register|Responders|Ajax|trigger|cancelBubble|stopPropagation|preventDefault|delete|readyState|complete|loaded|setInterval|WebKit|DOMContentLoaded|clearInterval|samp|kbd|dfn|label|caption|optgroup|colgroup|col|param|object|del|ins|bdo|link|script|acronym|abbr|button|address|dt|dl|dd|hr|br|cite|blockquote|option|fieldset|legend|textarea|input|form|li|ol|ul|h6|h5|h4|h3|h2|h1|code|pre|tfoot|tbody|thead|th|td|tr|table|img|em|strong|span|div|appendChild|createTextNode|createElement|MSIE|slice|nodeName|Function|cssText|htmlFor|className|class|replaceChild|replaceElement|inserBefore|insertAfter|removeChild|remove|previousElement|nextSibling|nextElement|Methods|Version'.split('|')))
Event.addBehavior({
    'img#leftCTA1:mouseover': function(e) {
        this.src = "/donbest/images/db/cta-real-time-odds-02.png";
        return false;
    }
});
Event.addBehavior({
    'img#leftCTA1:mouseout': function(e) {
        this.src = "/donbest/images/db/cta-real-time-odds-01.png";
        return false;
    }
});
Event.addBehavior({
    'img#leftCTA2:mouseover': function(e) {
        this.src = "/donbest/images/db/cta-express-odds-02.png";
        return false;
    }
});
Event.addBehavior({
    'img#leftCTA2:mouseout': function(e) {
        this.src = "/donbest/images/db/cta-express-odds-01.png";
        return false;
    }
});
Event.addBehavior({
    'img#leftCTA3:mouseover': function(e) {
        this.src = "/donbest/images/db/cta-guaranteed-picks-02.png";
        return false;
    }
});
Event.addBehavior({
    'img#leftCTA3:mouseout': function(e) {
        this.src = "/donbest/images/db/cta-guaranteed-picks.png";
        return false;
    }
})
Event.addBehavior({
    'img#leftCTA4:mouseover': function(e) {
        this.src = "/donbest/images/db/cta-rotation-schedule-02.png";
        return false;
    }
});
Event.addBehavior({
    'img#leftCTA4:mouseout': function(e) {
        this.src = "/donbest/images/db/cta-rotation-schedule.png";
        return false;
    }
});
var HtmlDdl = Class.create({
    holder: null,
    showTo: 0,
    initialize: function(divId) {
        this.holder = $(divId);
        this.output = this.holder.down('span');
        this.dd = this.holder.down('ul').hide();
        this.holder.selectedIndex = this.GetSelected();
        this.holder.options = {};
        this.BuildOptions();
        this.holder.observe('click', this.Show.bindAsEventListener(this));
        this.holder.observe('mouseout', this.Hide.bindAsEventListener(this));
        this.holder.observe('mouseover', this.Show.bindAsEventListener(this));
        this.dd.observe('mouseover', this.Show.bindAsEventListener(this));
    },
    GetSelected: function() {
        var ret = 0;
        this.holder.select('li').each(function(li, i) {
            if (li.hasClassName('ddl_itemselected'))
                ret = i;
        });
        return ret;
    },
    Show: function() {
        clearTimeout(this.showTo);
        this.dd.show();
    },
    Hide: function(val) {
        val = val == null ? 300 : val;
        this.showTo = setTimeout(function() {
            this.dd.hide();
        }.bind(this), val);
    },
    BuildOptions: function() {
        this.holder.select('li').each(function(li, i) {
            this.holder.options[i] = {
                text: li.down('a').innerHTML,
                value: li.down('a').rel
            };
        }.bind(this));
    },
    Select: function(obj, val) {
        $(obj.parentNode).siblings().each(function(li) {
            li.removeClassName('ddl_itemselected');
        });
        $(obj.parentNode).addClassName('ddl_itemselected');
        $(obj.parentNode.parentNode).hide();
        this.Update(obj.innerHTML);
        this.Hide(0);
        this.holder.selectedIndex = this.GetSelected();
        if ($(this.holder).getAttribute('onchange') != null) {
            eval('(function(){#{0}})()'.interpolate({
                0: $(this.holder).getAttribute('onchange')
            }));
        }
    },
    Update: function(val) {
        this.output.update(val);
    }
});
function livechatpopup(URL) {
    day = new Date();
    id = day.getTime();
    eval("page" + id + " = window.open(URL, '" + id + "','toolbar=0,scrollbars=0,location=0,statusbar=0,menubar=0,resizable=0,width=865,height=510');");
}
function popup(URL) {
    day = new Date();
    id = day.getTime();
    eval("page" + id + " = window.open(URL, '" + id + "','toolbar=0,scrollbars=1,location=0,statusbar=0,menubar=0,resizable=0,width=700,height=900');");
}
if (typeof(console) == 'undefined')
    window.console = {
        log: function(args) {}
    };
var MainMenu = Class.create({
    Options: {},
    VisibleKey: '',
    MainOptions: new Array(),
    SubOptions: new Array(),
    initialize: function(o) {
        this.Options = {};
        Object.extend(this.Options, o || {});
        this.InitMenu();
    }
});
MainMenu.MenuType = {
    MAINITEM: 1,
    SUBITEM: 2
}
MainMenu.prototype.InitMenu = function() {
    this.Holder = $(this.Options.menuId);
    this.Holder.down('ul').immediateDescendants().each(function(t, i) {
        var d = {
            obj: t,
            index: i,
            key: '#{0}_#{1}'.interpolate({
                0: this.Options.menuId,
                1: i
            }),
            hasMenu: t.down('ul') != null&&!t.hasClassName('menu_item_hide_dd'),
            link: t.down('a'),
            timeout: 0,
            dd: new Element('div', {
                style: 'display:none;'
            }).addClassName('mainMenuDDBox dropdownGeneral'),
            type: MainMenu.MenuType.MAINITEM,
            subOptions: new Array()
        };
        this.MainOptions.push(d);
        t.id = d.key;
    }.bind(this));
    Event.observe(window, 'load', function() {
        this.MainOptions.each(function(option) {
            if (!option.hasMenu)
                return;
            this.MakeDD(option, option.obj.down('ul'));
        }.bind(this));
    }.bind(this));
}
MainMenu.prototype.MakeDD = function(parent, obj) {
    parent.dd.appendChild(obj);
    $(document.body).appendChild(parent.dd);
    Event.observe(parent.obj, 'mouseover', function() {
        this.Show(parent);
    }.bind(this));
    Event.observe(parent.obj, 'mouseout', function() {
        this.Hide(parent);
    }.bind(this));
    Event.observe(parent.dd, 'mouseover', function() {
        this.Show(parent);
    }.bind(this));
    Event.observe(parent.dd, 'mouseout', function() {
        this.Hide(parent);
    }.bind(this));
    parent.dd.down('ul').immediateDescendants().each(function(t, i) {
        var pos = t.cumulativeOffset();
        var s = {
            obj: t,
            index: i,
            key: '#{0}_#{1}'.interpolate({
                0: parent.key,
                1: i
            }),
            hasMenu: t.down('ul') != null,
            link: t.down('a'),
            timeout: 0,
            dd: new Element('div', {
                style: 'display:none;'
            }).addClassName('mainMenuDDSubBox dropdownGeneral'),
            parent: parent,
            type: MainMenu.MenuType.SUBITEM
        };
        t.id = s.key;
        parent.subOptions.push(s);
    }.bind(this));
    parent.subOptions.each(function(option) {
        if (!option.hasMenu)
            return;
        this.MakeSubDD(option, option.obj.down('ul'));
    }.bind(this));
}
MainMenu.prototype.MakeSubDD = function(parent, obj) {
    parent.dd.appendChild(obj);
    $(document.body).appendChild(parent.dd);
    parent.link.appendChild(new Element('span').addClassName('dropdownHasSub').update(''));
    Event.observe(parent.obj, 'mouseover', function() {
        this.ShowSubMenu(parent);
    }.bind(this));
    Event.observe(parent.obj, 'mouseout', function() {
        this.Hide(parent);
    }.bind(this));
    Event.observe(parent.dd, 'mouseover', function() {
        this.ShowSubMenu(parent);
    }.bind(this));
    Event.observe(parent.dd, 'mouseout', function() {
        this.Hide(parent);
    }.bind(this));
    Event.observe(parent.dd, 'mouseover', function() {
        this.Show(parent.parent);
    }.bind(this));
    Event.observe(parent.dd, 'mouseout', function() {
        this.Hide(parent.parent);
    }.bind(this));
}
MainMenu.prototype.ShowSubMenu = function(p) {
    clearTimeout(p.timeout);
    if (p.key == this.VisibleKey)
        return;
    this.VisibleKey = parent.key;
    p.parent.subOptions.each(function(o) {
        $(o.dd).hide();
        $(o.obj).removeClassName('menu_item_selected');
    });
    var pos = p.pos;
    p.dd.setStyle({
        position: 'absolute',
        top: pos.top + 'px',
        left: (pos.left + p.dd.getWidth() + 3) + 'px'
    });
    p.obj.addClassName('menu_item_selected');
    p.dd.show();
}
MainMenu.prototype.Show = function(parent) {
    clearTimeout(parent.timeout);
    var invalid = parent.key.indexOf(this.VisibleKey) == 0&&!this.VisibleKey.empty();
    if (invalid)
        return;
    $(this.MainOptions).each(function(o) {
        if (o.key == parent.key)
            return;
        $(o.dd).hide();
        $(o.obj).removeClassName('menu_item_selected');
        if (o.type = MainMenu.MenuType.MAINITEM) {
            o.subOptions.each(function(s) {
                if (s.hasMenu)
                    s.dd.hide();
            }.bind(this));
        }
    }.bind(this));
    this.VisibleKey = parent.key;
    var pos = parent.obj.cumulativeOffset();
    parent.dd.setStyle({
        position: 'absolute',
        top: (pos.top + parent.obj.getHeight()) + 'px',
        left: (pos.left) + 'px'
    });
    parent.obj.addClassName('menu_item_selected');
    parent.dd.show();
    if (parent.type == MainMenu.MenuType.MAINITEM && parent.subOptions.length > 0) {
        parent.subOptions.each(function(s) {
            s.pos = s.obj.cumulativeOffset();
        });
    }
}
MainMenu.prototype.Hide = function(parent) {
    parent.timeout = setTimeout(function() {
        $(parent.dd).hide();
        this.VisibleKey = '';
        $(parent.obj).removeClassName('menu_item_selected');
    }.bind(this), 300);
}
var Menu = Class.create({
    initialize: function(options) {
        this.Options = options;
        this.Holder = $(options.holder);
        this.DropDowns = new Array();
        this.BindLinks(this);
    },
    BindLinks: function(g) {
        var i = 0;
        this.Holder.select('ul')[0].select('a').each(function(a) {
            if (a.select('span').length == 0)
                return;
            var sb = new StringBuilder();
            sb.Append('menu_');
            sb.Append(a.getAttribute('rel'));
            var dd = new DropDown({
                trigger: a.select('span')[0],
                baseElement: 'parent',
                displayElement: sb.ToString(),
                closeOnClick: false,
                align: 'right',
                shadow: {
                    opacity: .2,
                    offset: 4
                },
                effects: g.Options.effects,
                closeOnMouseout: true,
                padding: g.Options.padding,
                showOnMouseover: g.Options.showOnMouseover,
                onShowBegin: function() {
                    $(this.DropDowns).each(function(d) {
                        if (dd.Id != d.Id) {
                            d.HideDropDown();
                            d.BaseElement.removeClassName('menuBarSelected');
                        }
                    });
                    dd.BaseElement.addClassName('menuBarSelected');
                }.bind(this),
                onHideFinish: function() {
                    dd.BaseElement.removeClassName('menuBarSelected');
                }
            });
            dd.Id = 'dd_' + i;
            i++;
            this.DropDowns.push(dd);
        }.bind(this));
    }
});
var MessageBoard = function(vars) {
    var g = this;
    g.boxId = vars.boxId;
    g.typeId = vars.typeId;
    g.type = vars.type;
    g.page = vars.page;
    g.pageSize = vars.pageSize;
    g.locked = false;
    g.width = vars.width;
    this.initalize = function() {
        g.textBox = $('messageTextBox' + g.boxId);
        g.messageBox = $('messageBox' + g.boxId);
        g.hiddenHolder = $('hiddenBoxHolder');
        g.hiddenTextBox = $('messageTextBoxHidden');
        g.hiddenSubmit = $('submitReply');
    }
    this.changePage = function(newPage) {
        if (!g.locked) {
            g.locked = true;
            new Ajax.Request('/ajax/?a=Message_NewPage', {
                method: 'post',
                parameters: {
                    'page': newPage,
                    'type': g.type,
                    'typeId': g.typeId,
                    'width': g.width
                },
                onSuccess: function(t) {
                    g.page++;
                    $('messageHolder' + g.boxId).update(t.responseText);
                },
                onComplete: function(t) {
                    g.locked = false;
                }
            });
        }
    }
    this.addMessage = function(parent) {
        if (!g.locked) {
            g.locked = true;
            $(parent).innerHTML = 'Sending ...';
            if (g.textBox.value == '') {
                g.locked = false;
                return true;
            }
            new Ajax.Request('/ajax/?a=Message_Submit', {
                method: 'post',
                parameters: {
                    'text': g.textBox.value,
                    'reply': 0,
                    'type': g.type,
                    'typeId': g.typeId
                },
                onSuccess: function(t) {
                    var msg = t.responseText.evalJSON(false).messages.message[0];
                    g.printMessage(g.messageBox, msg, 'new');
                    g.textBox.value = '';
                },
                onComplete: function() {
                    g.locked = false;
                    $(parent).innerHTML = "Send";
                }
            });
        }
    }
    this.replyMessage = function(replyId) {
        var parent = $('message' + replyId);
        parent.appendChild(g.hiddenHolder);
        g.hiddenSubmit.innerHTML = 'Reply';
        g.hiddenHolder.show();
        g.hiddenTextBox.value = '';
        g.hiddenSubmit.onclick = function() {
            if (!g.locked) {
                g.locked = true;
                if (g.hiddenTextBox.value == '') {
                    g.locked = false;
                    return false;
                }
                new Ajax.Request('/ajax/?a=Message_Submit', {
                    method: 'post',
                    parameters: {
                        'text': g.hiddenTextBox.value,
                        'reply': replyId,
                        'type': g.type,
                        'typeId': g.typeId
                    },
                    onSuccess: function(t) {
                        var msg = t.responseText.evalJSON(false).messages.message[0];
                        g.printMessage(parent, msg, 'reply');
                        g.hiddenTextBox.value = '';
                    },
                    onComplete: function() {
                        g.locked = false;
                        g.hiddenHolder.hide();
                    }
                });
            }
            return false;
        }
    }
    this.printMessage = function(p, msg, type) {
        var temp = $('messageTemplate').cloneNode(true);
        temp.select('.mUsername')[0].innerHTML = msg.username;
        temp.select('.mTime')[0].innerHTML = msg.elapsedTime;
        temp.select('.mMessage')[0].innerHTML = msg.text;
        temp.select('.mReply')[0].onclick = function() {
            g.replyMessage(msg.id);
        };
        temp.select('.mRemove')[0].onclick = function() {
            g.removeMessage(msg.id);
        };
        temp.id = 'message' + msg.id;
        if (type == 'reply')
            p.appendChild(temp);
        else
            p.insertBefore(temp, p.firstChild);
        temp.toggle('slow');
    }
    this.removeMessage = function(id) {
        if (!g.locked) {
            g.locked = true;
            new Ajax.Request('/ajax/?a=Message_Remove', {
                method: 'post',
                parameters: {
                    'id': id,
                    'type': g.type,
                    'typeId': g.typeId
                },
                onSuccess: function(t) {
                    var msg = $('message' + id).hide();
                },
                onComplete: function() {
                    g.locked = false;
                    g.hiddenHolder.hide();
                }
            });
        }
    }
    this.editMessage = function(id) {
        var parent = $('message' + id);
        var message = $('messageText' + id);
        g.hiddenTextBox.value = message.select('span')[0].innerHTML;
        g.hiddenSubmit.update('Edit');
        message.appendChild(g.hiddenHolder);
        g.hiddenHolder.show();
        g.hiddenSubmit.onclick = function() {
            if (!g.locked) {
                g.locked = true;
                g.hiddenSubmit.update('Sending ...');
                new Ajax.Request('/ajax/?a=Message_Edit', {
                    method: 'post',
                    parameters: {
                        'id': id,
                        'type': g.type,
                        'typeId': g.typeId,
                        'text': g.hiddenTextBox.value
                    },
                    onSuccess: function(t) {
                        message.select('span')[0].innerHTML = t.responseText;
                    },
                    onComplete: function() {
                        g.hiddenSubmit.update('Edit');
                        g.locked = false;
                        g.hiddenHolder.hide();
                    }
                });
            }
            return false;
        };
    }
}
var Module = {
    SendingAjax: false,
    Sortable: function(columnId) {
        Sortable.create('col_' + columnId, {
            tag: 'div',
            only: 'module',
            handle: 'moduleControls',
            overlap: 'horizontal',
            constraint: false,
            onUpdate: function() {
                this.UpdateModuleOrder(columnId);
            }.bind(this)
        });
    },
    UpdateModuleOrder: function(columnId) {
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        var modules = new Array();
        $('col_' + columnId).select('div.module').each(function(div) {
            modules.push(div.getAttribute('moduleId'));
        });
        new Ajax.Request('/ajax/?a=Modules_UpdateOrder', {
            method: 'post',
            parameters: {
                column: columnId,
                pageId: _PAGEID,
                leagueId: _LEAGUEID,
                ids: modules.join(',')
            },
            onComplete: function(t) {
                this.SendingAjax = false;
            }.bind(this)
        });
    },
    Delete: function(obj, id) {
        if (!confirm('Are you sure you want to delete this module?'))
            return;
        if (this.SendingAjax)
            return;
        this.SendingAjax = false;
        new Ajax.Request('/ajax/?a=Modules_Delete', {
            method: 'post',
            parameters: {
                pageId: _PAGEID,
                uniqueId: id
            },
            onSuccess: function(t) {
                var div = $$('div[moduleid="#{0}"]'.interpolate({
                    0: id
                }))[0]
                new Effect.BlindUp(div, {
                    duration: .3,
                    afterFinish: function() {
                        div.remove();
                    }.bind(this)
                });
            },
            onComplete: function(t) {
                this.Sending = false;
            }.bind(this)
        });
    },
    ModulesConfigured: {},
    ModuleList: null,
    GetModuleList: function(options) {
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        new Ajax.Request('/ajax/?a=Modules_GetModuleList', {
            method: 'post',
            parameters: null,
            onComplete: function(t) {
                this.SendingAjax = false;
                if (options.onComplete)
                    options.onComplete();
            }.bind(this),
            onSuccess: function(t) {
                this.ModuleList = t.responseJSON.root;
            }.bind(this)
        });
    },
    BindModuleList: function(obj, column, dd) {
        obj.update('');
        var table = new Element('table', {
            cellspacing: '0',
            cellpadding: '0'
        });
        var tbody;
        if (Prototype.Browser.IE) {
            tbody = new Element('tbody');
            table.appendChild(tbody);
        } else
            tbody = table;
        var row = new Element('tr');
        tbody.appendChild(row);
        var cell;
        var itemCount = Math.ceil(this.ModuleList.length / 4);
        $(this.ModuleList).each(function(module, i) {
            if ((i)%itemCount == 0) {
                cell = new Element('td')
                row.appendChild(cell);
            }
            var a = new Element('a', {
                href: 'javascript:void(0);'
            });
            Event.observe(a, 'click', function() {
                if (this.SendingAjax)
                    return;
                this.SendingAjax = true;
                new Ajax.Request('/ajax/?a=Modules_AddModule', {
                    method: 'post',
                    parameters: {
                        moduleId: module.id,
                        pageId: _PAGEID,
                        column: column
                    },
                    onComplete: function(t) {
                        this.SendingAjax = false;
                        this.UpdateModuleOrder(column);
                    }.bind(this),
                    onSuccess: function(t) {
                        var html = t.responseJSON.html;
                        var holder = new Element('div').update(html).firstDescendant();
                        var columnObj = $('col_' + column);
                        columnObj.firstDescendant().insert({
                            after: holder
                        });
                        var i = 1;
                        columnObj.select('div.module').each(function(div) {
                            div.id = 'module#{0}_#{1}'.interpolate({
                                0: column,
                                1: i
                            });
                            i++;
                        });
                        Sortable.destroy('col_' + column);
                        this.Sortable(column);
                    }.bind(this)
                });
            }.bind(this))
            a.update(module.name);
            cell.appendChild(a);
        }.bind(this));
        dd.Shadow.obj.setStyle({
            height: dd.DisplayElement.getHeight() + 'px'
        });
        obj.appendChild(table);
    },
    ShowOptions: function(obj, column, dd) {
        obj = $(obj);
        if (this.ModuleList == null)
            this.ModuleList = this.GetModuleList({
                onComplete: function() {
                    this.BindModuleList(obj, column, dd);
                }.bind(this)
            });
        else
            this.BindModuleList(obj, column, dd);
    },
    HideWarning: function(id) {
        if ($('moduleWarning' + id))
            new Effect.BlindUp($('moduleWarning' + id), {
                duration: .2
            });
    },
    ShowWarning: function(module) {},
    ShowConfigureBackEnd: function(obj, moduleId, id, pageId)
    {
        _PAGEID = pageId;
        this.ShowConfigure(obj, moduleId, id);
    },
    ShowConfigure: function(obj, moduleId, id) {
        var divRef = $('moduleData' + id);
        if (divRef != null)
        {
            this.ModulesConfigured['module' + id] = $('moduleData' + id).innerHTML;
        } else
        {
            this.ModulesConfigured['module' + id] = $('_divAdminSettingsByModule').innerHTML;
        }
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        new Ajax.Request('/ajax/?a=Modules_GetConfigure', {
            method: 'post',
            parameters: {
                moduleId: moduleId,
                uniqueId: id,
                leagueId: _LEAGUEID,
                pageId: _PAGEID
            },
            onSuccess: function(t) {
                var module = t.responseJSON.module;
                Module.HideWarning(module.uniqueId);
                var submit = new Element('div').addClassName('moduleSubmit alignCenter');
                var saveButton = new Element('input', {
                    id: 'moduleSaveButton' + module.uniqueId,
                    type: 'button',
                    value: 'Save Settings',
                    style: 'margin-right: 5px;'
                }).addClassName('adminFormButtonSmall');
                Event.observe(saveButton, 'click', function(e) {
                    this.ModuleSave(saveButton, module.moduleId, module.uniqueId);
                }.bindAsEventListener(this));
                submit.appendChild(saveButton);
                var cancelButton = new Element('input', {
                    id: 'moduleCancelButton' + module.uniqueId,
                    type: 'button',
                    value: 'Cancel'
                }).addClassName('adminFormButtonSmall');
                Event.observe(cancelButton, 'click', function(e) {
                    this.ModuleCancel(cancelButton, module.moduleId, module.uniqueId);
                }.bindAsEventListener(this));
                submit.appendChild(cancelButton);
                var jsscript = '<script language="javascript" type="text/javascript" src="#{0}module/#{1}/#{2}/#{3}/"></script>'.interpolate({
                    0: _HERE.replace('/donbest', '/'),
                    1: _PAGEID,
                    2: moduleId,
                    3: id
                });
                submit.appendChild(new Element('a', {
                    href: 'javascript:void(0);',
                    'onclick': "$('outputScript_#{0}').toggle()".interpolate({
                        0: id
                    })
                }).update('<br /><strong>Get Script</strong>'));
                submit.appendChild(new Element('input', {
                    'type': 'text',
                    id: 'outputScript_#{0}'.interpolate({
                        0: id
                    }),
                    'value': jsscript,
                    'onfocus': 'this.select();'
                }).addClassName('txt').setStyle({
                    fontSize: '10px',
                    width: '100%',
                    border: '2px solid #2699E5',
                    color: '#0D73BB',
                    display: 'none',
                    marginTop: '5px'
                }));
                if (divRef != null)
                {
                    $('moduleData' + module.uniqueId).addClassName('moduleConfigureDiv');
                }
                this.SetModule(module, false, submit);
            }.bind(this),
            onComplete: function(t) {
                this.SendingAjax = false;
            }.bind(this)
        });
    },
    ModuleSave: function(obj, moduleId, id) {
        if (this.SendingAjax)
            return;
        this.SendingAjax = false;
        obj.disabled = true;
        obj.value = 'Saving ...';
        var m = $('moduleData' + id);
        if (m == null)
        {
            m = $('_divAdminSettingsByModule');
        }
        var vars = SBRForum.FormValues(m);
        var ret = {
            vars: vars.serialize,
            moduleId: moduleId,
            uniqueId: id,
            pageId: _PAGEID
        }
        new Ajax.Request('/ajax/?a=Modules_SaveModule', {
            method: 'post',
            parameters: ret,
            onSuccess: function(t) {
                var module = t.responseJSON.module;
                if (module.status == '400')
                    this.ShowWarning(module);
                else {
                    var moduleHolder = $('moduleData' + id);
                    if (moduleHolder != null)
                    {
                        this.HideWarning(id);
                        this.SetModule(module, true, null);
                        moduleHolder.removeClassName('moduleConfigureDiv');
                    }
                }
            }.bind(this),
            onComplete: function(t) {
                this.SendingAjax = false;
                obj.disabled = false;
                obj.value = "Save Settings";
            }.bind(this)
        });
    },
    ModuleCancel: function(obj, moduleId, id) {
        var m = $('moduleData' + id);
        if (m != null)
        {
            m.parentNode.select('div.moduleControls')[0].setStyle({
                display: ''
            });
            m.update(this.ModulesConfigured['module' + id]);
            m.removeClassName('moduleConfigureDiv');
        } else
        {
            m = $('_divAdminSettingsByModule');
            m.update(this.ModulesConfigured['module' + id]);
        }
    },
    SetModule: function(module, showConfig, button) {
        if (module.status != '304') {
            var m = $('moduleData' + module.uniqueId);
            if (m != null)
            {
                m.parentNode.select('div.moduleControls')[0].setStyle({
                    display: 'none'
                });
                m.update(module.innerHtml);
                if (showConfig)
                    m.parentNode.select('div.moduleControls')[0].setStyle({
                        display: ''
                    });
                if (button)
                    m.appendChild(button);
                m.innerHTML.evalScripts();
            } else
            {
                m = $('_divAdminSettingsByModule');
                m.update(module.innerHtml);
                if (button)
                    m.appendChild(button);
                m.innerHTML.evalScripts();
            }
        }
    }
}
function moveModule(obj, dir) {
    var section = null, holder = null;
    $(obj).ancestors().each(function(d) {
        if (d.className == 'section' && section == null)
            section = d;
    });
    holder = $(section).ancestors()[0];
    var ppp = $(section).getStyle('position');
    var www = $(section).getStyle('width');
    var move;
    if (dir == 'up')
        move = $(section).previousSiblings();
    else
        move = $(section).nextSiblings();
    if (move.length == 0)
        return;
    if (!ie6) {
        var pos1 = $(move[0]).cumulativeOffset();
        var pos2 = $(section).cumulativeOffset();
        $(section).setStyle({
            zIndex: 100
        });
        $(move[0]).setStyle({
            zIndex: 99
        });
        if (dir == 'down') {}
        var placeHolder = null;
        new Effect.Move(section, {
            x: 0,
            y: dir == "down" ? $(move[0]).getHeight(): pos1.top - pos2.top,
            duration: .5,
            beforeStart: function() {
                if (dir == 'down') {
                    $(section).setStyle({
                        position: 'absolute',
                        top: pos2.top + 'px',
                        left: pos2.left - 10 + 'px',
                        width: $(holder).getWidth() + 'px'
                    });
                    placeHolder = new Element('div').setStyle({
                        width: $(holder).getWidth() + 'px',
                        height: $(section).getHeight() + 25 + 'px',
                        'float': 'left'
                    });
                    holder.insertBefore(move[0], section);
                    holder.insertBefore(placeHolder, section);
                }
            },
            afterFinish: function() {
                if (dir == 'up')
                    holder.insertBefore(section, move[0]);
                if (placeHolder)
                    holder.removeChild(placeHolder);
                section.setStyle({
                    top: null,
                    zIndex: null,
                    left: null,
                    position: ppp,
                    width: www
                });
                move[0].setStyle({
                    zIndex: null
                });
                getModuleOrder(holder.id, section.getAttribute('moduleid'));
            }
        });
    } else {
        if (dir == 'up')
            holder.insertBefore(section, move[0]);
        else
            holder.insertBefore(move[0], section);
        getModuleOrder(holder.id, section.getAttribute('moduleid'));
    }
}
function updateColumnOrder()
{
    var columns = $('pageColumn').select('.sortColumn');
    var columnsResult = '';
    var limit = columns.length;
    for (index = 0; index < limit; index++)
    {
        if (columnsResult.length > 0)
        {
            columnsResult += ",";
        }
        columnsResult += $(columns[index]).getAttribute("columnId");
    }
    new Ajax.Request('/ajax/?a=Modules_updateColumnOrder', {
        method: 'post',
        parameters: {
            columns: columnsResult,
            pageId: _PAGEID
        },
        onSuccess: function(req) {}
    });
}
var col = new Array();
function getModuleOrder(holderId, unique) {
    var columnSplit = holderId.split('_');
    var columnId = columnSplit[1];
    var holder = $(holderId);
    var output = '';
    $(holderId).select('div.section').each(function(div) {
        output += div.getAttribute('moduleid') + ",";
    });
    if (output.indexOf(',')==-1) {
        return;
    }
    output = output.substring(0, output.length - 1);
    var t = output.split(',');
    if (t.indexOf(unique)==-1) {
        return;
    }
    col[holderId] = output;
    new Ajax.Request('/ajax/?a=Modules_updateColumn', {
        method: 'post',
        parameters: {
            ids: output,
            column: columnId,
            unique: unique,
            pageId: _PAGEID,
            leagueId: _LEAGUEID
        },
        requestHeaders: {
            Accept: 'application/json'
        },
        onSuccess: function(t) {
            var module = t.responseText.evalJSON(false).module;
            setModule(module, module.showConfig, null);
        }
    });
}
function redoColumns() {
    var maxHeight = 0;
    try {
        $(columns).each(function(l) {
            var i = 1;
            if (Prototype.Browser.IE)
                $(l).setStyle({
                    height: 'auto'
                });
            $(l).select('div.section').each(function(div) {
                div.id = l + '_module_' + i;
                i++;
            });
            maxHeight = $(l).getHeight() > maxHeight ? $(l).getHeight() : maxHeight;
        });
        if (Prototype.Browser.IE) {
            $(columns).each(function(l) {
                $(l).setStyle({
                    height: maxHeight + 'px'
                });
            });
        }
    } catch (e) {}
}
function createSortable(columns, columnId) {
    var cols = $(columns);
    var colLength = cols.length;
    cols.each(function(c) {
        var i = 1;
        $(c).select('div.section').each(function(div) {
            div.id = c + '_module_' + i;
            i++;
        });
        var uniqueId = 0;
        var sc = Sortable.create(c, {
            tag: 'div',
            only: 'section',
            handle: 'modMove',
            containment: columns,
            dropOnEmpty: true,
            constraint: false,
            onStart: function(a) {
                if (document.selection)
                    document.selection.empty();
                else
                    window.getSelection().removeAllRanges();
            },
            onEnd: function(a) {
                if (document.selection)
                    document.selection.empty();
                else
                    window.getSelection().removeAllRanges();
            },
            onUpdate: function(t) {
                getModuleOrder(c, uniqueId);
                redoColumns();
            },
            onChange: function(e) {
                uniqueId = e.getAttribute('moduleid');
            }
        });
    });
}
var Overlay = {
    Hide: function(options) {
        var o = {
            onclick: function() {
                return;
            }
        };
        Object.extend(o, options || {});
        var overlay = Overlay.Get(o);
        if (o.onclick != null)
            o.onclick(overlay);
        new Effect.Fade(overlay, {
            duration: .3
        });
    },
    Get: function(options) {
        var o = {
            onclick: function() {
                return;
            }
        };
        Object.extend(o, options || {});
        var overlay;
        if ($('overlay') == null) {
            overlay = new Element('div', {
                'class': 'overlay',
                id: 'overlay'
            });
            overlay.className = 'overlay';
            overlay.setStyle({
                height: document.viewport.getHeight() + 'px',
                width: document.viewport.getWidth() + 'px'
            });
            $(document.body).appendChild(overlay);
            Event.observe(window, 'resize', function() {
                overlay.setStyle({
                    height: document.viewport.getHeight() + 'px',
                    width: document.viewport.getWidth() + 'px'
                });
            });
            Event.observe(overlay, 'click', function() {
                Overlay.Hide(o);
            });
        } else
            overlay = $('overlay');
        return overlay;
    }
}
var SBRForum = {
    SendingAjax: false,
    FormValues: function(obj) {
        var vars = {};
        obj.select('input:not([type="checkbox"])').each(function(i) {
            if (i.name || i.id) {
                if (i.getAttribute('type') == 'radio' && i.checked)
                    vars[i.name ? i.name: i.id] = i.value;
                else if (i.getAttribute('type') != 'radio')
                    vars[i.name ? i.name: i.id] = escape(i.value);
            }
        });
        obj.select('select').each(function(s) {
            if (s.multiple)
            {
                var count = 0;
                var valuesSelected = {};
                var lengthOpt = s.options.length;
                for (var i = 0; i < lengthOpt; i++)
                {
                    var opt = s.options[i];
                    if (opt.selected)
                    {
                        if (s.name || s.id)
                        {
                            valuesSelected[s.name ? s.name + count: s.id + count] = escape(opt.value);
                            count++;
                        }
                    }
                }
                if (s.name || s.id)
                {
                    var vals = "";
                    for (va in valuesSelected) {
                        if (vals.length > 0)
                        {
                            vals += ",";
                        }
                        vals += valuesSelected[va];
                    }
                    var finishValues = {
                        valuesSelected: valuesSelected,
                        serialize: vals
                    };
                    vars[s.name ? s.name: s.id] = finishValues.serialize;
                }
            } else
            {
                if (s.name || s.id) {
                    vars[s.name ? s.name: s.id] = s.selectedIndex!=-1 ? escape(s.options[s.selectedIndex].value) : "";
                }
            }
        });
        obj.select('textarea').each(function(t) {
            if (t.name || t.id) {
                vars[t.name ? t.name: t.id] = escape(t.value);
            }
        });
        obj.select('input[type="checkbox"]').each(function(c) {
            if (c.name || c.id) {
                vars[c.name ? c.name: c.id] = c.checked;
            }
        });
        var s = "";
        for (k in vars) {
            s += k.replace(/.+\$.+\$/, '') + '=' + vars[k] + "&";
        }
        return {
            vars: vars,
            serialize: s
        };
    },
    ToggleHeadline: function(boxId, idx) {
        var p = $('headlines-content');
        p.update($(boxId).innerHTML);
        var tabs = $('headline-tabs').select('a');
        for (var i = 0; i < tabs.length; i++) {
            var tab = tabs[i];
            if (idx == i)
                $(tab).addClassName('headline-tabs-selected');
            else
                $(tab).removeClassName('headline-tabs-selected');
        }
        var more = $('more-button');
        var league = $('league');
        more.href = league.innerHTML == 'ALL' ? '/' + boxId + '/all.aspx' : '/' + boxId + '/' + league.innerHTML + '/all.aspx';
        more.update('More ' + (league.innerHTML == 'ALL' ? '' : league.innerHTML + ' ') + boxId);
    },
    FixPageForm: function() {
        Event.observe(window, 'load', function() {
            if ($('aspnetForm'))
                $('aspnetForm').action = window.location;
        });
    },
    Quiz: {
        SelectAnswer: function(obj) {
            $(obj.name + '_holderDiv').select('img').each(function(img) {
                img.src = _HERE + '/images/spacer.gif';
                img.parentNode.className = '';
            });
            var img = $(obj.id + '_imgLeft');
            img.src = _HERE + '/images/quizSelect-Left.gif';
            img.parentNode.className = 'quizSelected';
        }
    },
    SearchOptions: {
        Location: 0
    },
    SetSearchLocation: function(obj, loc) {
        $('searchTextBox').setStyle({
            backgroundPosition: 'right -' + (loc==-1 ? '10000' : (161 + 15 * loc)) + 'px'
        });
        SBRForum.SearchOptions.Location = loc;
    },
    Search: function() {
        var search = escape($('searchTextBox').value.replace(/ /ig, '-').toLowerCase());
        window.location = "/Sports+Betting/#{0}.aspx?loc=#{1}".interpolate({
            0: search,
            1: SBRForum.SearchOptions.Location
        });
    },
    CreateEnterActions: function() {
        Event.observe(window, 'load', function() {
            $(document.body).select('input[type="text"]', 'input[type="password"]').each(function(textbox) {
                if (textbox.getAttribute('buttonRel') != null) {
                    var buttonId = textbox.getAttribute('buttonRel');
                    Event.observe(textbox, 'keydown', function(event) {
                        if (event.which || event.keyCode) {
                            if ((event.which == 13) || (event.keyCode == 13)) {
                                $(buttonId).click();
                                Event.stop(event);
                            }
                        } else
                            Event.stop(event);
                    }.bind(this));
                }
            });
        });
    },
    TextboxPretext: function() {
        Event.observe(window, 'load', function() {
            $(document.body).select('input[type="text"]').each(function(textbox) {
                if (textbox.getAttribute('preText') != null) {
                    var preText = textbox.getAttribute('preText');
                    textbox.value = preText;
                    Event.observe(textbox, 'focus', function(event) {
                        if (textbox.value == preText) {
                            textbox.value = '';
                        }
                    });
                    Event.observe(textbox, 'blur', function(event) {
                        if (textbox.value == '') {
                            textbox.value = preText;
                        }
                    });
                }
            });
        });
    }
}
SBRForum.FixPageForm();
SBRForum.CreateEnterActions();
SBRForum.TextboxPretext();
var Scoreboard = {
    SendingAjax: false,
    MakeCalendar: function(objId, startLeague, startDate, startLeagueId) {
        Event.observe(window, 'load', function() {
            this.calendar = new Calendar({
                holderId: objId,
                startDate: startDate,
                calendarClass: 'scoreboardCal',
                showPrevious: true,
                showNext: true,
                nextText: '►',
                prevText: '◄',
                onDateSelect: function(date) {
                    if (startLeagueId == 1) {
                        window.location.href = '/Betting-Odds/#{0}/'.interpolate({
                            0: dateFormat(date.date, "yyyymmdd")
                        });
                    } else {
                        window.location.href = '/Scores/#{0}+Odds/#{1}.aspx'.interpolate({
                            0: startLeague.replace(' ', '+'),
                            1: dateFormat(date.date, "yyyymmdd")
                        });
                    }
                },
                onCreate: function() {},
                onNext: function() {},
                onPrev: function() {}
            });
        }.bind(this));
    },
    ToggleDay: function(obj, holderId) {
        var pNode = $(obj.parentNode);
        pNode.select('a').each(function(a) {
            a.removeClassName('sbfp_selected');
        });
        $(obj).addClassName('sbfp_selected');
        $('sbfb_holder').select('div.sbfb_day').each(function(div) {
            div.setStyle({
                display: 'none'
            });
        });
        $(holderId).setStyle({
            display: 'block'
        });
    },
    DeleteMonthScoresCache: function(obj, date) {
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        var message = $(obj).select('span')[0];
        message.update('Deleting Cache ...');
        this.DeleteCache('month', 0, date, 0, message);
    },
    DeleteScoresCache: function(obj, leagueId, date) {
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        var message = $(obj).select('span')[0];
        message.update('Updating Cache ... ');
        this.DeleteCache('scores', leagueId, date, 0, message);
    },
    DeleteLinesCache: function(obj, leagueId, eventId, type) {
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        var message = $(obj).select('span')[0];
        message.update('Updating Cache ... ');
        this.DeleteCache('lines', leagueId, type, eventId, message);
    },
    DeleteCache: function(type, leagueId, param, eventId, messageHolder) {
        new Ajax.Request('/ajax/?a=Scoreboard_DeleteCache', {
            method: 'post',
            parameters: {
                type: type,
                league: leagueId,
                param: param,
                eventId: eventId
            },
            onSuccess: function(t) {
                var data = t.responseJSON;
                if (data.status == 200) {
                    $(data.id).update(data.html);
                    if (type == "month")
                        messageHolder.update('Cache Deleted.');
                    else
                        messageHolder.update('Cache Updated.');
                } else {
                    messageHolder.update('Error Ocurred.');
                }
                setTimeout(function() {
                    messageHolder.update('');
                }, 2000);
            }.bind(this),
            onError: function(t) {
                messageHolder.update('Error Ocurred.');
            },
            onComplete: function(t) {
                this.SendingAjax = false;
            }.bind(this)
        });
    },
    Cache: {},
    GetFrontPageSB: function(leagueName) {
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        if (this.Cache[leagueName]) {
            var timestamp = new Date();
            if (timestamp.setSeconds(this.Cache[leagueName].timestamp.getSeconds() + 30) > new Date()) {
                $('fp_scoreboard_data').update(this.Cache[leagueName].innerHTML);
                this.SendingAjax = false;
                return;
            }
        }
        new Ajax.Request('/ajax/?a=[CMS]Scoreboard_ChangeLeague', {
            method: 'post',
            parameters: {
                league: leagueName,
                location: 'frontpage'
            },
            onSuccess: function(t) {
                this.Cache[leagueName] = {
                    timestamp: new Date(),
                    innerHTML: t.responseText
                }
                $('fp_scoreboard_data').update(t.responseText);
            }.bind(this),
            onComplete: function(t) {
                this.SendingAjax = false;
            }.bind(this)
        });
    }
}
var Statistic = Class.create({
    options: {},
    SendingAjax: false,
    _DDL_TIME_ZONES: '',
    _CookiName: '',
    _Div_Load: '',
    _SELECTED_LINEID: '0',
    TIME_REFRESH: 25000,
    firstTime: true,
    initialize: function(options) {
        this.options = {
            League: 'All',
            LeagueId: '-1',
            CurrentTZ: '',
            TZSelected: '',
            XSLTPath: '',
            feedId: 0,
            every: 30,
            datepart: 'minutes',
            at: '12:00',
            refreshValue: '',
            date: new Date().format('mm/dd/yyyy'),
            timestamp: 0,
            bookId: 2,
            view: ''
        }
        Object.extend(this.options, options || {});
    },
    ChangeTimeZone: function(type) {
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        var divLoadingBull = $(this._Div_Load);
        divLoadingBull.style.visibility = "visible";
        divLoadingBull.style.display = "block";
        var ddlTimeZone = $(this._DDL_TIME_ZONES);
        var timeZoneSelected = ddlTimeZone.options[ddlTimeZone.selectedIndex].text;
        var timeZomeValue = ddlTimeZone.options[ddlTimeZone.selectedIndex].value;
        this.setTimeZoneCooki(this._CookiName, timeZomeValue, '', '/', '', '');
        this.options.TZSelected = timeZoneSelected;
        var extraOptions = {
            Type: type,
            LineType: this._SELECTED_LINEID
        };
        Object.extend(extraOptions, this.options || {});
        new Ajax.Request('/ajax/?a=[DonbestCommon]Statistic_ChangeTimeZone', {
            method: 'post',
            parameters: extraOptions,
            requestHeaders: {
                Accept: 'application/json'
            },
            onSuccess: function(t) {
                var divOut = $("_DivOutput");
                var html = t.responseJSON.html;
                divOut.update(html);
            }.bind(this),
            onComplete: function(t) {
                this.SendingAjax = false;
                divLoadingBull.style.visibility = "hidden";
                divLoadingBull.style.display = "none";
            }.bind(this)
        });
        return false;
    },
    setTimeZoneCooki: function(name, value, expires, path, domain, secure) {
        var expires_date = new Date();
        expires_date.setYear(expires_date.getYear() + 1);
        document.cookie = name + "=" + value + ";path=" + path;
        return false;
    },
    GetOddsByLineType: function(allOdds) {
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        var divLoadingBull = $(this._Div_Load);
        divLoadingBull.style.visibility = "visible";
        divLoadingBull.style.display = "block";
        var ddlTimeZone = $(this._DDL_TIME_ZONES);
        var timeZoneSelected = ddlTimeZone.options[ddlTimeZone.selectedIndex].text;
        var timeZomeValue = ddlTimeZone.options[ddlTimeZone.selectedIndex].value;
        this.options.TZSelected = timeZoneSelected;
        var extraOptions = {
            LineType: this._SELECTED_LINEID
        };
        Object.extend(extraOptions, this.options || {});
        if (allOdds == true) {
            new Ajax.Request('/ajax/?a=[DonbestCommon]Statistic_AllChangeLineOdds', {
                method: 'post',
                parameters: extraOptions,
                requestHeaders: {
                    Accept: 'application/json'
                },
                onSuccess: function(t) {
                    var divOut = $("_DivOutput");
                    var html = t.responseJSON.html;
                    divOut.update(html);
                }.bind(this),
                onComplete: function(t) {
                    this.SendingAjax = false;
                    divLoadingBull.style.visibility = "hidden";
                    divLoadingBull.style.display = "none";
                }.bind(this)
            });
        } else {
            new Ajax.Request('/ajax/?a=[DonbestCommon]Statistic_ChangeLineOdds', {
                method: 'post',
                parameters: extraOptions,
                requestHeaders: {
                    Accept: 'application/json'
                },
                onSuccess: function(t) {
                    var divOut = $("_DivOutput");
                    var html = t.responseJSON.html;
                    divOut.update(html);
                }.bind(this),
                onComplete: function(t) {
                    this.SendingAjax = false;
                    divLoadingBull.style.visibility = "hidden";
                    divLoadingBull.style.display = "none";
                }.bind(this)
            });
        }
        return false;
    },
    ChangeLineOdds: function(button, allOdds) {
        if (this.options._Selected_TabId != button.id) {
            button.parentNode.className = "tabItem tabItemSelect"
            var oldSelectedTab = $(this.options._Selected_TabId);
            oldSelectedTab.parentNode.className = "tabItem";
            oldSelectedTab = null;
            this.options._Selected_TabId = button.id;
            if (button.innerHTML == 'Spreads/Totals' || button.innerHTML == 'ML/Totals') {
                this._SELECTED_LINEID = 0;
            }
            if (button.innerHTML == 'Money Lines') {
                this._SELECTED_LINEID = 2;
            }
            if (button.innerHTML == 'Run Lines' || button.innerHTML == 'Puck Lines' || button.innerHTML == 'Spreads') {
                this._SELECTED_LINEID = 1;
            }
            if (button.innerHTML == 'Totals') {
                this._SELECTED_LINEID = 3;
            }
            this.GetOddsByLineType(allOdds);
        }
        return false;
    },
    ChangeDay: function(pLeagueName, pLeagueId, pNewDate) {
        var date = pNewDate.split('/');
        var url = window.location.protocol + "//" + window.location.hostname + (window.location.port ? ":" + window.location.port : "") + window.location.pathname;
        if (url.lastIndexOf('/') + 1 == url.length) {
            url = url.substring(0, url.lastIndexOf('/')) + "/" + date[2] + date[0] + date[1] + ".html";
            window.location = url;
        } else {
            var currentTime = new Date();
            var month = currentTime.getMonth() + 1;
            var day = currentTime.getDate();
            var year = currentTime.getFullYear();
            var dateNow = year + "" + (month > 9 ? month : "0" + month) + "" + (day > 9 ? day : "0" + day);
            var dateNew = date[2] + date[0] + date[1];
            if (dateNow == dateNew) {
                window.location = url.substring(0, url.lastIndexOf('/') + 1);
            } else {
                window.location = url.substring(0, url.lastIndexOf('/') + 1) + date[2] + date[0] + date[1] + ".html";
            }
        }
    },
    UpdateLeague: function(obj, leagueId, leagueName) {
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        var divLoadingBull = $(this._Div_Load);
        divLoadingBull.style.visibility = "visible";
        divLoadingBull.style.display = "block";
        var ddlTimeZone = $(this._DDL_TIME_ZONES);
        var timeZoneSelected = ddlTimeZone.options[ddlTimeZone.selectedIndex].text;
        this.options.TZSelected = timeZoneSelected;
        this.options.League = leagueName;
        this.options.LeagueId = leagueId;
        new Ajax.Request('/ajax/?a=[DonbestCommon]Statistic_UpdateLeague', {
            method: 'post',
            parameters: this.options,
            onSuccess: function(t) {
                var data = t.responseJSON;
                obj = $(obj);
                $(obj.parentNode).select('li').each(function(li) {
                    li.removeClassName('tabItemSelect');
                });
                obj.addClassName('tabItemSelect');
                $('_DivOutput').update(data.html);
            }.bind(this),
            onComplete: function(t) {
                this.SendingAjax = false;
                divLoadingBull.style.visibility = "hidden";
                divLoadingBull.style.display = "none";
            }.bind(this)
        });
    },
    UpdateLines: function(league, ts, date) {
        return;
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        new Ajax.Request('/ajax/?a=[DonbestCommon]Statistic_UpdateLines', {
            method: 'post',
            parameters: {
                league: league,
                ts: ts,
                date: date,
                bookId: this.options.bookId
            },
            onSuccess: function(t) {
                var data = t.responseJSON;
                var listEvents = data.LineChanges.Games;
                this.options.timestamp = data.LineChanges.Timestamp;
                var lineAtt;
                switch (league) {
                case 5:
                case 7:
                    lineAtt = "Money";
                    break;
                default:
                    lineAtt = "Spread";
                }
                for (var i = 0; i < listEvents.length; i = i + 3) {
                    var divLine1 = $('_Div_Line_' + league + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_" + 2);
                    var divLine2 = $('_Div_Line_' + league + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_" + 2);
                    var divScore1 = $('_Div_Score_' + league + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId);
                    var divScore2 = $('_Div_Score_' + league + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId);
                    var divTime1 = $('_Div_Time_' + league + "_" + listEvents[i].EventId + "_up");
                    var divTime2 = $('_Div_Time_' + league + "_" + listEvents[i].EventId + "_down");
                    if (divLine1 != null && divLine2 != null) {
                        var BooksTeam1 = listEvents[i].Books;
                        var BooksTeam2 = listEvents[i + 1].Books;
                        var moneyTeam1;
                        var moneyTeam2;
                        var spreadTeam1;
                        var spreadTeam2;
                        var totalTeam1;
                        var totalTeam2;
                        for (var j = 0; j < BooksTeam1.length; j++) {
                            if (BooksTeam1[j].BookId == "2") {
                                moneyTeam1 = BooksTeam1[j].Money;
                                spreadTeam1 = BooksTeam1[j].Spread;
                                totalTeam1 = BooksTeam1[j].Total;
                            }
                        }
                        for (var z = 0; z < BooksTeam2.length; z++) {
                            if (BooksTeam2[z].BookId == "2") {
                                moneyTeam2 = BooksTeam2[z].Money;
                                spreadTeam2 = BooksTeam2[z].Spread;
                                totalTeam2 = BooksTeam2[z].Total;
                            }
                        }
                        if (lineAtt == "Money") {
                            if (moneyTeam1 && moneyTeam2 && moneyTeam1 < moneyTeam2) {
                                divLine1.update(moneyTeam1);
                                if (totalTeam2)
                                    divLine2.update(totalTeam2);
                            } else {
                                if (moneyTeam1 && moneyTeam2) {
                                    divLine2.update(moneyTeam2);
                                    if (totalTeam1)
                                        divLine1.update(totalTeam1);
                                }
                            }
                        } else {
                            if (spreadTeam1 && spreadTeam2 && spreadTeam1 < spreadTeam2) {
                                divLine1.update(spreadTeam1);
                                if (totalTeam2)
                                    divLine2.update(totalTeam2);
                            } else {
                                if (spreadTeam1 && spreadTeam2) {
                                    divLine2.update(spreadTeam2);
                                    if (totalTeam1)
                                        divLine1.update(totalTeam1);
                                }
                            }
                        }
                    }
                    if (divScore1 != null) {
                        if (listEvents[i + 2].period && listEvents[i + 2].homeScore && listEvents[i + 2].awayScore) {
                            if (listEvents[i + 2].period == "Final") {
                                divScore1.update(this.BoldWinner(listEvents[i + 2].homeScore, listEvents[i + 2].awayScore, listEvents[i + 2].homeScore));
                            } else {
                                divScore1.update(listEvents[i + 2].homeScore);
                            }
                        } else {
                            if (listEvents[i + 2].homeScore)
                                divScore1.update(listEvents[i + 2].homeScore);
                        }
                    }
                    if (divScore2 != null) {
                        if (listEvents[i + 2].period && listEvents[i + 2].homeScore && listEvents[i + 2].awayScore) {
                            if (listEvents[i + 2].period == "Final") {
                                divScore2.update(this.BoldWinner(listEvents[i + 2].awayScore, listEvents[i + 2].homeScore, listEvents[i + 2].awayScore));
                            } else {
                                divScore2.update(listEvents[i + 2].awayScore);
                            }
                        } else {
                            if (listEvents[i + 2].awayScore)
                                divScore2.update(listEvents[i + 2].awayScore);
                        }
                    }
                    if (divTime1 != null) {
                        if (listEvents[i + 2].status && listEvents[i + 2].status != "0" && listEvents[i + 2].period != "Final")
                            divTime1.update(listEvents[i + 2].status);
                    }
                    if (divTime2 != null) {
                        if (listEvents[i + 2].period && listEvents[i + 2].period == "Final") {
                            divTime1.update('&nbsp');
                            divTime2.update('Final');
                        } else {
                            if (listEvents[i + 2].period != "0")
                                divTime2.update(listEvents[i + 2].period);
                        }
                    }
                }
            }.bind(this),
            onComplete: function(t) {
                this.SendingAjax = false;
            }.bind(this)
        });
    },
    UpdateLinesByType: function(league, ts, date) {
        return;
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        new Ajax.Request('/ajax/?a=[DonbestCommon]Statistic_UpdateLines', {
            method: 'post',
            parameters: {
                league: league,
                ts: ts,
                date: date
            },
            onSuccess: function(t) {
                var data = t.responseJSON;
                var listEvents = data.LineChanges.Games;
                this.options.timestamp = data.LineChanges.Timestamp;
                for (var i = 0; i < listEvents.length; i = i + 3) {
                    var divPitcher1;
                    var divPitcher2;
                    if (league == 5) {
                        divPitcher1 = $('_Div_Pitcher_' + league + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId);
                        divPitcher2 = $('_Div_Pitcher_' + league + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId);
                        if (divPitcher1 && divPitcher2) {
                            if (listEvents[i].Pitcher)
                                divPitcher1.update(listEvents[i].Pitcher.Name + "-" + listEvents[i].Pitcher.Hand);
                            if (listEvents[i + 1].Pitcher)
                                divPitcher2.update(listEvents[i + 1].Pitcher.Name + "-" + listEvents[i + 1].Pitcher.Hand);
                        }
                    }
                    var divScore1 = $('_Div_Score_' + league + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId);
                    var divScore2 = $('_Div_Score_' + league + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId);
                    if (divScore1 != null) {
                        if (listEvents[i + 2].period && listEvents[i + 2].homeScore && listEvents[i + 2].awayScore) {
                            if (listEvents[i + 2].period == "Final") {
                                divScore1.update("<b>" + listEvents[i + 2].homeScore + "</b>");
                            } else {
                                divScore1.update(listEvents[i + 2].homeScore);
                            }
                        } else {
                            if (listEvents[i + 2].homeScore)
                                divScore1.update(listEvents[i + 2].homeScore);
                        }
                    }
                    if (divScore2 != null) {
                        if (listEvents[i + 2].period && listEvents[i + 2].homeScore && listEvents[i + 2].awayScore) {
                            if (listEvents[i + 2].period == "Final") {
                                divScore2.update("<b>" + listEvents[i + 2].awayScore + "</b>");
                            } else {
                                divScore2.update(listEvents[i + 2].awayScore);
                            }
                        } else {
                            if (listEvents[i + 2].awayScore)
                                divScore2.update(listEvents[i + 2].awayScore);
                        }
                    }
                    var divStatus = $("_Div_Status_" + league + "_" + listEvents[i].EventId);
                    var divPeriod = $("_Div_Period_" + league + "_" + listEvents[i].EventId);
                    if (divStatus != null && divPeriod != null) {
                        if (listEvents[i + 2].period == "Final") {
                            divStatus.update("&nbsp");
                            divPeriod.update("FINAL");
                        } else {
                            if (listEvents[i + 2].status && listEvents[i + 2].status != "0")
                                divStatus.update(listEvents[i + 2].status);
                            if (listEvents[i + 2].period && listEvents[i + 2].period != "0")
                                divPeriod.update(listEvents[i + 2].period);
                        }
                    }
                    var BooksTeam1 = listEvents[i].Books;
                    var BooksTeam2 = listEvents[i + 1].Books;
                    if (this._SELECTED_LINEID == "0") {
                        var lineAtt;
                        switch (league) {
                        case 5:
                        case 7:
                            lineAtt = "Money";
                            break;
                        default:
                            lineAtt = "Spread";
                        }
                        var BooksTeam1 = listEvents[i].Books;
                        var BooksTeam2 = listEvents[i + 1].Books;
                        var moneyTeam1;
                        var moneyTeam2;
                        var spreadTeam1;
                        var spreadTeam2;
                        var totalTeam1;
                        var totalTeam2;
                        for (var j = 0; j < BooksTeam1.length; j++) {
                            var divLine1 = $('_Div_Line_' + league + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_" + this._SELECTED_LINEID + "_" + BooksTeam1[j].BookId);
                            var divLine2 = $('_Div_Line_' + league + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_" + this._SELECTED_LINEID + "_" + BooksTeam2[j].BookId);
                            if (divLine1 != null && divLine2 != null) {
                                moneyTeam1 = BooksTeam1[j].Money;
                                spreadTeam1 = BooksTeam1[j].Spread;
                                totalTeam1 = BooksTeam1[j].Total;
                                moneyTeam2 = BooksTeam2[j].Money;
                                spreadTeam2 = BooksTeam2[j].Spread;
                                totalTeam2 = BooksTeam2[j].Total;
                                if (lineAtt == "Money") {
                                    if (moneyTeam1 && moneyTeam2 && moneyTeam1 < moneyTeam2) {
                                        divLine1.update(this.FractionIt(moneyTeam1));
                                        if (totalTeam2)
                                            divLine2.update(this.FractionIt(totalTeam2));
                                    } else {
                                        if (moneyTeam1 && moneyTeam2) {
                                            divLine2.update(this.FractionIt(moneyTeam2));
                                            if (totalTeam1)
                                                divLine1.update(this.FractionIt(totalTeam1));
                                        }
                                    }
                                } else {
                                    if (spreadTeam1 && spreadTeam2 && spreadTeam1 < spreadTeam2) {
                                        divLine1.update(this.FractionIt(spreadTeam1));
                                        if (totalTeam2)
                                            divLine2.update(this.FractionIt(totalTeam2));
                                    } else {
                                        if (spreadTeam1 && spreadTeam2) {
                                            divLine2.update(this.FractionIt(spreadTeam2));
                                            if (totalTeam1)
                                                divLine1.update(this.FractionIt(totalTeam1));
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        for (var books1 = 0; books1 < BooksTeam1.length; books1++) {
                            var divLine1 = $('_Div_Line_' + league + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_" + this._SELECTED_LINEID + "_" + BooksTeam1[books1].BookId);
                            if (divLine1 != null) {
                                switch (this._SELECTED_LINEID) {
                                case 1:
                                    if (BooksTeam1[books1].Spread && BooksTeam1[books1].Price) {
                                        if (BooksTeam1[books1].Spread == "" || BooksTeam1[books1].Spread == 0 || BooksTeam1[books1].Price == 0) {
                                            divLine1.update("-");
                                        } else {
                                            divLine1.update(this.FractionIt(BooksTeam1[books1].Spread) + " " + BooksTeam1[books1].Price);
                                        }
                                    }
                                    break;
                                case 2:
                                    if (BooksTeam1[books1].Money) {
                                        if (!BooksTeam1[books1].Money || BooksTeam1[books1].Money == "" || BooksTeam1[books1].Money == "0") {
                                            divLine1.update("-");
                                        } else {
                                            divLine1.update(BooksTeam1[books1].Money);
                                        }
                                    }
                                    break;
                                case 3:
                                    if (BooksTeam1[books1].Total && BooksTeam1[books1].TotalPrice) {
                                        if (BooksTeam1[books1].Total == "" || BooksTeam1[books1].Total == 0 || BooksTeam1[books1].TotalPrice == "" || BooksTeam1[books1].TotalPrice == 0) {
                                            divLine1.update("-");
                                        } else {
                                            var valueTotal = new Element('div', {
                                                'class': 'oddsAlignMiddleTwo'
                                            });
                                            valueTotal.update(this.FractionIt(BooksTeam1[books1].Total));
                                            divLine1.update(valueTotal);
                                            divLine1.innerHTML += " " + BooksTeam1[books1].TotalPrice;
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                        for (var books2 = 0; books2 < BooksTeam2.length; books2++) {
                            var divLine2 = $('_Div_Line_' + league + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_" + this._SELECTED_LINEID + "_" + BooksTeam2[books2].BookId);
                            if (divLine2 != null) {
                                switch (this._SELECTED_LINEID) {
                                case 1:
                                    if (BooksTeam2[books2].Spread && BooksTeam2[books2].Price) {
                                        if (BooksTeam2[books2].Spread == "" || BooksTeam2[books2].Spread == 0 || BooksTeam2[books2].Price == 0) {
                                            divLine2.update("-");
                                        } else {
                                            divLine2.update(this.FractionIt(BooksTeam2[books2].Spread) + " " + BooksTeam2[books2].Price);
                                        }
                                    }
                                    break;
                                case 2:
                                    if (BooksTeam2[books2].Money) {
                                        if (!BooksTeam2[books2].Money || BooksTeam2[books2].Money == "" || BooksTeam2[books2].Money == "0") {
                                            divLine2.update("-");
                                        } else {
                                            divLine2.update(BooksTeam2[books2].Money);
                                        }
                                    }
                                    break;
                                case 3:
                                    if (BooksTeam2[books2].Total && BooksTeam2[books2].TotalPrice) {
                                        if (BooksTeam2[books2].Total == "" || BooksTeam2[books2].Total == 0 || BooksTeam2[books2].TotalPrice == "" || BooksTeam2[books2].TotalPrice == 0) {
                                            divLine2.update("-");
                                        } else {
                                            divLine2.update(BooksTeam2[books2].TotalPrice);
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }.bind(this),
            onComplete: function(t) {
                this.SendingAjax = false;
            }.bind(this)
        });
    },
    UpdateLinesScores: function(league, ts, date) {
        return;
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        new Ajax.Request('/ajax/?a=[DonbestCommon]Statistic_UpdateLines', {
            method: 'post',
            parameters: {
                league: league,
                ts: ts,
                date: date
            },
            onSuccess: function(t) {
                var data = t.responseJSON;
                var listEvents = data.LineChanges.Games;
                this.options.timestamp = data.LineChanges.Timestamp;
                var lineAtt;
                switch (league) {
                case 5:
                case 7:
                    lineAtt = "Money";
                    break;
                default:
                    lineAtt = "Spread";
                }
                for (var i = 0; i < listEvents.length; i = i + 3) {
                    var divPitcher1;
                    var divPitcher2;
                    if (league == 5) {
                        divPitcher1 = $('_Div_Pitcher_' + league + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId);
                        divPitcher2 = $('_Div_Pitcher_' + league + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId);
                        if (divPitcher1 && divPitcher2) {
                            if (listEvents[i].Pitcher)
                                divPitcher1.update(listEvents[i].Pitcher.Name + "-" + listEvents[i].Pitcher.Hand);
                            if (listEvents[i + 1].Pitcher)
                                divPitcher2.update(listEvents[i + 1].Pitcher.Name + "-" + listEvents[i + 1].Pitcher.Hand);
                        }
                    }
                    var ScoresTeam1 = listEvents[i].Scores;
                    var ScoresTeam2 = listEvents[i + 1].Scores;
                    var s1t1;
                    var s2t1;
                    var s3t1;
                    var s4t1;
                    var s5t1;
                    var s6t1;
                    var s7t1;
                    var s8t1;
                    var s9t1;
                    var s1t2;
                    var s2t2;
                    var s3t2;
                    var s4t2;
                    var s5t2;
                    var s6t2;
                    var s7t2;
                    var s8t2;
                    var s9t2;
                    if (ScoresTeam1) {
                        if (ScoresTeam1[0])
                            s1t1 = ScoresTeam1[0].Value;
                        if (ScoresTeam1[1])
                            s2t1 = ScoresTeam1[1].Value;
                        if (ScoresTeam1[2])
                            s3t1 = ScoresTeam1[2].Value;
                        if (ScoresTeam1[3])
                            s4t1 = ScoresTeam1[3].Value;
                        if (ScoresTeam1[4])
                            s5t1 = ScoresTeam1[4].Value;
                        if (ScoresTeam1[5])
                            s6t1 = ScoresTeam1[5].Value;
                        if (ScoresTeam1[6])
                            s7t1 = ScoresTeam1[6].Value;
                        if (ScoresTeam1[7])
                            s8t1 = ScoresTeam1[7].Value;
                        if (ScoresTeam1[8])
                            s9t1 = ScoresTeam1[8].Value;
                        for (var st1 = 0; st1 < ScoresTeam1.length; st1++) {
                            var AwayScore = $('_Div_Score_' + league + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_" + ScoresTeam1[st1].Id);
                            if (AwayScore && ScoresTeam1[st1].Value)
                                AwayScore.update(ScoresTeam1[st1].Value);
                        }
                    }
                    if (ScoresTeam2) {
                        if (ScoresTeam2[0])
                            s1t2 = ScoresTeam2[0].Value;
                        if (ScoresTeam2[1])
                            s2t2 = ScoresTeam2[1].Value;
                        if (ScoresTeam2[2])
                            s3t2 = ScoresTeam2[2].Value;
                        if (ScoresTeam2[3])
                            s4t2 = ScoresTeam2[3].Value;
                        if (ScoresTeam2[4])
                            s5t2 = ScoresTeam2[4].Value;
                        if (ScoresTeam2[5])
                            s6t2 = ScoresTeam2[5].Value;
                        if (ScoresTeam2[6])
                            s7t2 = ScoresTeam2[6].Value;
                        if (ScoresTeam2[7])
                            s8t2 = ScoresTeam2[7].Value;
                        if (ScoresTeam2[8])
                            s9t2 = ScoresTeam2[8].Value;
                        for (var st2 = 0; st2 < ScoresTeam2.length; st2++) {
                            var HomeScore = $('_Div_Score_' + league + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_" + ScoresTeam2[st2].Id);
                            if (HomeScore && ScoresTeam2[st2].Value)
                                HomeScore.update(ScoresTeam2[st2].Value);
                        }
                    }
                    switch (league) {
                    case 5:
                        {
                            var fiveScores1 = $('_Div_Score_' + league + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_1st5");
                            if (fiveScores1) {
                                if (s1t1 && s2t1 && s3t1 && s4t1 && s5t1)
                                    fiveScores1.update(parseInt(s1t1) + parseInt(s2t1) + parseInt(s3t1) + parseInt(s4t1) + parseInt(s5t1));
                            }
                            var fiveScores2 = $('_Div_Score_' + league + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_1st5");
                            if (fiveScores2) {
                                if (s1t2 && s2t2 && s3t2 && s4t2 && s5t2)
                                    fiveScores2.update(parseInt(s1t2) + parseInt(s2t2) + parseInt(s3t2) + parseInt(s4t2) + parseInt(s5t2));
                            }
                            break;
                        }
                    case 6:
                        {
                            var fiveScores11 = $('_Div_Score_' + league + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_1st");
                            if (fiveScores11) {
                                if (s1t1)
                                    fiveScores11.update(parseInt(s1t1));
                            }
                            var fiveScores21 = $('_Div_Score_' + league + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_1st");
                            if (fiveScores21) {
                                if (s1t2)
                                    fiveScores21.update(parseInt(s1t2));
                            }
                            var fiveScores12 = $('_Div_Score_' + league + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_2nd");
                            if (fiveScores12) {
                                if (s2t1)
                                    fiveScores12.update(parseInt(s2t1));
                            }
                            var fiveScores22 = $('_Div_Score_' + league + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_2nd");
                            if (fiveScores22) {
                                if (s2t2)
                                    fiveScores22.update(parseInt(s2t2));
                            }
                            break;
                        }
                    case 7:
                    case 2:
                    case 3:
                        {
                            var fiveScores11 = $('_Div_Score_' + league + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_1st");
                            if (fiveScores11) {
                                if (s1t1 && s2t1)
                                    fiveScores11.update(parseInt(s1t1) + parseInt(s2t1));
                            }
                            var fiveScores21 = $('_Div_Score_' + league + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_1st");
                            if (fiveScores21) {
                                if (s1t2 && s2t2)
                                    fiveScores21.update(parseInt(s1t2) + parseInt(s2t2));
                            }
                            var fiveScores12 = $('_Div_Score_' + league + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_2nd");
                            if (fiveScores12) {
                                if (s3t1 && s4t1)
                                    fiveScores12.update(parseInt(s3t1) + parseInt(s4t1));
                            }
                            var fiveScores22 = $('_Div_Score_' + league + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_2nd");
                            if (fiveScores22) {
                                if (s3t2 && s4t2)
                                    fiveScores22.update(parseInt(s3t2) + parseInt(s4t2));
                            }
                            break;
                        }
                    }
                    var FinalScore1 = $('_Div_Score_' + league + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_final");
                    var FinalScore2 = $('_Div_Score_' + league + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_final");
                    if (FinalScore1 && FinalScore2) {
                        if ((listEvents[i + 2].status != '' || listEvents[i + 2].status != '0') && (listEvents[i + 2].period != '' || listEvents[i + 2].period != '0')) {
                            if (listEvents[i + 2].awayScore)
                                FinalScore1.update(listEvents[i + 2].awayScore);
                            if (listEvents[i + 2].homeScore)
                                FinalScore2.update(listEvents[i + 2].homeScore);
                        }
                    }
                    var divLine1 = $('_Div_Line_' + league + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_" + listEvents[i].BookOpenerId);
                    var divLine2 = $('_Div_Line_' + league + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_" + listEvents[i + 1].BookOpenerId);
                    if (divLine1 != null && divLine2 != null) {
                        var BooksTeam1 = listEvents[i].Books;
                        var BooksTeam2 = listEvents[i + 1].Books;
                        var moneyTeam1;
                        var moneyTeam2;
                        var spreadTeam1;
                        var spreadTeam2;
                        var totalTeam1;
                        var totalTeam2;
                        for (var j = 0; j < BooksTeam1.length; j++) {
                            if (BooksTeam1[j].BookId == listEvents[i].BookOpenerId) {
                                moneyTeam1 = BooksTeam1[j].Money;
                                spreadTeam1 = BooksTeam1[j].Spread;
                                totalTeam1 = BooksTeam1[j].Total;
                            }
                        }
                        for (var z = 0; z < BooksTeam2.length; z++) {
                            if (BooksTeam2[z].BookId == listEvents[i].BookOpenerId) {
                                moneyTeam2 = BooksTeam2[z].Money;
                                spreadTeam2 = BooksTeam2[z].Spread;
                                totalTeam2 = BooksTeam2[z].Total;
                            }
                        }
                        if (lineAtt == "Money") {
                            if (moneyTeam1 && moneyTeam2 && moneyTeam1 < moneyTeam2) {
                                if (moneyTeam1 == "0") {
                                    divLine1.update("PK");
                                } else {
                                    divLine1.update(moneyTeam1);
                                }
                                if (totalTeam2)
                                    divLine2.update(totalTeam2);
                            } else {
                                if (moneyTeam1 && moneyTeam2) {
                                    if (moneyTeam2 == "0") {
                                        divLine2.update("PK");
                                    } else {
                                        divLine2.update(moneyTeam2);
                                    }
                                    if (totalTeam1)
                                        divLine1.update(totalTeam1);
                                }
                            }
                        } else {
                            if (spreadTeam1 && spreadTeam2 && spreadTeam1 < spreadTeam2) {
                                if (spreadTeam1 == "0") {
                                    divLine1.update("PK");
                                } else {
                                    divLine1.update(spreadTeam1);
                                }
                                if (totalTeam2)
                                    divLine2.update(totalTeam2);
                            } else {
                                if (spreadTeam1 && spreadTeam2) {
                                    if (spreadTeam2 == "0") {
                                        divLine2.update("PK");
                                    } else {
                                        divLine2.update(spreadTeam2);
                                    }
                                    if (totalTeam1)
                                        divLine1.update(totalTeam1);
                                }
                            }
                        }
                    }
                    var divStatus = $("_Div_Status_" + league + "_" + listEvents[i].EventId);
                    var divPeriod = $("_Div_Period_" + league + "_" + listEvents[i].EventId);
                    if (divStatus != null && divPeriod != null) {
                        if (listEvents[i + 2].period == "Final") {
                            divStatus.update("&nbsp");
                            divPeriod.update("FINAL");
                        } else {
                            if (listEvents[i + 2].status && listEvents[i + 2].status != "0")
                                divStatus.update(listEvents[i + 2].status);
                            if (listEvents[i + 2].period && listEvents[i + 2].period != "0")
                                divPeriod.update(listEvents[i + 2].period);
                        }
                    }
                }
            }.bind(this),
            onComplete: function(t) {
                this.SendingAjax = false;
            }.bind(this)
        });
    },
    UpdateLinesAllOdds: function(league, ts, date) {
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        new Ajax.Request('/ajax/?a=[DonbestCommon]Statistic_UpdateLines', {
            method: 'post',
            parameters: {
                league: league,
                ts: ts,
                date: date
            },
            onSuccess: function(t) {
                var data = t.responseJSON;
                var listLeagues = data.LineChanges.Leagues;
                for (var lid = 0; lid < listLeagues.length; lid++) {
                    var listEvents = listLeagues[lid].Games;
                    var leagueId = listLeagues[lid].LeagueId;
                    this.options.timestamp = data.LineChanges.Timestamp;
                    for (var i = 0; i < listEvents.length; i = i + 3) {
                        var divPitcher1;
                        var divPitcher2;
                        if (league == 5) {
                            divPitcher1 = $('_Div_Pitcher_' + leagueId + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId);
                            divPitcher2 = $('_Div_Pitcher_' + leagueId + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId);
                            if (divPitcher1 && divPitcher2) {
                                if (listEvents[i].Pitcher)
                                    divPitcher1.update(listEvents[i].Pitcher.Name + "-" + listEvents[i].Pitcher.Hand);
                                if (listEvents[i + 1].Pitcher)
                                    divPitcher2.update(listEvents[i + 1].Pitcher.Name + "-" + listEvents[i + 1].Pitcher.Hand);
                            }
                        }
                        var divScore1 = $('_Div_Score_' + leagueId + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId);
                        var divScore2 = $('_Div_Score_' + leagueId + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId);
                        if (divScore1 != null) {
                            if (listEvents[i + 2].period && listEvents[i + 2].homeScore && listEvents[i + 2].awayScore) {
                                if (listEvents[i + 2].period == "Final") {
                                    divScore1.update("<b>" + listEvents[i + 2].homeScore + "</b>");
                                } else {
                                    divScore1.update(listEvents[i + 2].homeScore);
                                }
                            } else {
                                if (listEvents[i + 2].homeScore)
                                    divScore1.update(listEvents[i + 2].homeScore);
                            }
                        }
                        if (divScore2 != null) {
                            if (listEvents[i + 2].period && listEvents[i + 2].homeScore && listEvents[i + 2].awayScore) {
                                if (listEvents[i + 2].period == "Final") {
                                    divScore2.update("<b>" + listEvents[i + 2].awayScore + "</b>");
                                } else {
                                    divScore2.update(listEvents[i + 2].awayScore);
                                }
                            } else {
                                if (listEvents[i + 2].awayScore)
                                    divScore2.update(listEvents[i + 2].awayScore);
                            }
                        }
                        var divStatus = $("_Div_Status_" + leagueId + "_" + listEvents[i].EventId);
                        var divPeriod = $("_Div_Period_" + leagueId + "_" + listEvents[i].EventId);
                        if (divStatus != null && divPeriod != null) {
                            if (listEvents[i + 2].period == "Final") {
                                divStatus.update("&nbsp");
                                divPeriod.update("FINAL");
                            } else {
                                if (listEvents[i + 2].status && listEvents[i + 2].status != "0")
                                    divStatus.update(listEvents[i + 2].status);
                                if (listEvents[i + 2].period && listEvents[i + 2].period != "0")
                                    divPeriod.update(listEvents[i + 2].period);
                            }
                        }
                        var BooksTeam1 = listEvents[i].Books;
                        var BooksTeam2 = listEvents[i + 1].Books;
                        if (this._SELECTED_LINEID == "0") {
                            var lineAtt;
                            switch (leagueId) {
                            case"5":
                            case"7":
                                lineAtt = "Money";
                                break;
                            default:
                                lineAtt = "Spread";
                            }
                            var BooksTeam1 = listEvents[i].Books;
                            var BooksTeam2 = listEvents[i + 1].Books;
                            var moneyTeam1;
                            var moneyTeam2;
                            var spreadTeam1;
                            var spreadTeam2;
                            var totalTeam1;
                            var totalTeam2;
                            for (var j = 0; j < BooksTeam1.length; j++) {
                                var divLine1 = $('_Div_Line_' + leagueId + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_" + this._SELECTED_LINEID + "_" + BooksTeam1[j].BookId);
                                var divLine2 = $('_Div_Line_' + leagueId + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_" + this._SELECTED_LINEID + "_" + BooksTeam2[j].BookId);
                                if (divLine1 != null && divLine2 != null) {
                                    moneyTeam1 = BooksTeam1[j].Money;
                                    spreadTeam1 = BooksTeam1[j].Spread;
                                    totalTeam1 = BooksTeam1[j].Total;
                                    moneyTeam2 = BooksTeam2[j].Money;
                                    spreadTeam2 = BooksTeam2[j].Spread;
                                    totalTeam2 = BooksTeam2[j].Total;
                                    if (lineAtt == "Money") {
                                        if (moneyTeam1 && moneyTeam2 && moneyTeam1 < moneyTeam2) {
                                            divLine1.update(this.FractionIt(moneyTeam1));
                                            if (totalTeam2)
                                                divLine2.update(this.FractionIt(totalTeam2));
                                        } else {
                                            if (moneyTeam1 && moneyTeam2) {
                                                divLine2.update(this.FractionIt(moneyTeam2));
                                                if (totalTeam1)
                                                    divLine1.update(this.FractionIt(totalTeam1));
                                            }
                                        }
                                    } else {
                                        if (spreadTeam1 && spreadTeam2 && spreadTeam1 < spreadTeam2) {
                                            divLine1.update(this.FractionIt(spreadTeam1));
                                            if (totalTeam2)
                                                divLine2.update(this.FractionIt(totalTeam2));
                                        } else {
                                            if (spreadTeam1 && spreadTeam2) {
                                                divLine2.update(this.FractionIt(spreadTeam2));
                                                if (totalTeam1)
                                                    divLine1.update(this.FractionIt(totalTeam1));
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            for (var books1 = 0; books1 < BooksTeam1.length; books1++) {
                                var divLine1 = $('_Div_Line_' + leagueId + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_" + this._SELECTED_LINEID + "_" + BooksTeam1[books1].BookId);
                                if (divLine1 != null) {
                                    switch (this._SELECTED_LINEID) {
                                    case 1:
                                        if (BooksTeam1[books1].Spread && BooksTeam1[books1].Price) {
                                            if (BooksTeam1[books1].Spread == "" || BooksTeam1[books1].Spread == 0 || BooksTeam1[books1].Price == 0) {
                                                divLine1.update("-");
                                            } else {
                                                divLine1.update(this.FractionIt(BooksTeam1[books1].Spread) + " " + BooksTeam1[books1].Price);
                                            }
                                        }
                                        break;
                                    case 2:
                                        if (BooksTeam1[books1].Money) {
                                            if (!BooksTeam1[books1].Money || BooksTeam1[books1].Money == "" || BooksTeam1[books1].Money == "0") {
                                                divLine1.update("-");
                                            } else {
                                                divLine1.update(BooksTeam1[books1].Money);
                                            }
                                        }
                                        break;
                                    case 3:
                                        if (BooksTeam1[books1].Total && BooksTeam1[books1].TotalPrice) {
                                            if (BooksTeam1[books1].Total == "" || BooksTeam1[books1].Total == 0 || BooksTeam1[books1].TotalPrice == "" || BooksTeam1[books1].TotalPrice == 0) {
                                                divLine1.update("-");
                                            } else {
                                                var valueTotal = new Element('div', {
                                                    'class': 'oddsAlignMiddleTwo'
                                                });
                                                valueTotal.update(this.FractionIt(BooksTeam1[books1].Total));
                                                divLine1.update(valueTotal);
                                                divLine1.innerHTML += " " + BooksTeam1[books1].TotalPrice;
                                            }
                                        }
                                        break;
                                    }
                                }
                            }
                            for (var books2 = 0; books2 < BooksTeam2.length; books2++) {
                                var divLine2 = $('_Div_Line_' + leagueId + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_" + this._SELECTED_LINEID + "_" + BooksTeam2[books2].BookId);
                                if (divLine2 != null) {
                                    switch (this._SELECTED_LINEID) {
                                    case 1:
                                        if (BooksTeam2[books2].Spread && BooksTeam2[books2].Price) {
                                            if (BooksTeam2[books2].Spread == "" || BooksTeam2[books2].Spread == 0 || BooksTeam2[books2].Price == 0) {
                                                divLine2.update("-");
                                            } else {
                                                divLine2.update(this.FractionIt(BooksTeam2[books2].Spread) + " " + BooksTeam2[books2].Price);
                                            }
                                        }
                                        break;
                                    case 2:
                                        if (BooksTeam2[books2].Money) {
                                            if (!BooksTeam2[books2].Money || BooksTeam2[books2].Money == "" || BooksTeam2[books2].Money == "0") {
                                                divLine2.update("-");
                                            } else {
                                                divLine2.update(BooksTeam2[books2].Money);
                                            }
                                        }
                                        break;
                                    case 3:
                                        if (BooksTeam2[books2].Total && BooksTeam2[books2].TotalPrice) {
                                            if (BooksTeam2[books2].Total == "" || BooksTeam2[books2].Total == 0 || BooksTeam2[books2].TotalPrice == "" || BooksTeam2[books2].TotalPrice == 0) {
                                                divLine2.update("-");
                                            } else {
                                                divLine2.update(BooksTeam2[books2].TotalPrice);
                                            }
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }.bind(this),
            onComplete: function(t) {
                this.SendingAjax = false;
            }.bind(this)
        });
    },
    UpdateLinesAllScores: function(league, ts, date) {
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        new Ajax.Request('/ajax/?a=[DonbestCommon]Statistic_UpdateLines', {
            method: 'post',
            parameters: {
                league: league,
                ts: ts,
                date: date
            },
            onSuccess: function(t) {
                var data = t.responseJSON;
                var listLeagues = data.LineChanges.Leagues;
                for (var lid = 0; lid < listLeagues.length; lid++) {
                    var listEvents = listLeagues[lid].Games;
                    var leagueId = listLeagues[lid].LeagueId;
                    this.options.timestamp = data.LineChanges.Timestamp;
                    var lineAtt;
                    switch (leagueId) {
                    case"5":
                    case"7":
                        lineAtt = "Money";
                        break;
                    default:
                        lineAtt = "Spread";
                    }
                    for (var i = 0; i < listEvents.length; i = i + 3) {
                        var divPitcher1;
                        var divPitcher2;
                        if (leagueId == 5) {
                            divPitcher1 = $('_Div_Pitcher_' + leagueId + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId);
                            divPitcher2 = $('_Div_Pitcher_' + leagueId + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId);
                            if (divPitcher1 && divPitcher2) {
                                if (listEvents[i].Pitcher)
                                    divPitcher1.update(listEvents[i].Pitcher.Name + "-" + listEvents[i].Pitcher.Hand);
                                if (listEvents[i + 1].Pitcher)
                                    divPitcher2.update(listEvents[i + 1].Pitcher.Name + "-" + listEvents[i + 1].Pitcher.Hand);
                            }
                        }
                        var ScoresTeam1 = listEvents[i].Scores;
                        var ScoresTeam2 = listEvents[i + 1].Scores;
                        var s1t1;
                        var s2t1;
                        var s3t1;
                        var s4t1;
                        var s5t1;
                        var s6t1;
                        var s7t1;
                        var s8t1;
                        var s9t1;
                        var s1t2;
                        var s2t2;
                        var s3t2;
                        var s4t2;
                        var s5t2;
                        var s6t2;
                        var s7t2;
                        var s8t2;
                        var s9t2;
                        if (ScoresTeam1) {
                            if (ScoresTeam1[0])
                                s1t1 = ScoresTeam1[0].Value;
                            if (ScoresTeam1[1])
                                s2t1 = ScoresTeam1[1].Value;
                            if (ScoresTeam1[2])
                                s3t1 = ScoresTeam1[2].Value;
                            if (ScoresTeam1[3])
                                s4t1 = ScoresTeam1[3].Value;
                            if (ScoresTeam1[4])
                                s5t1 = ScoresTeam1[4].Value;
                            if (ScoresTeam1[5])
                                s6t1 = ScoresTeam1[5].Value;
                            if (ScoresTeam1[6])
                                s7t1 = ScoresTeam1[6].Value;
                            if (ScoresTeam1[7])
                                s8t1 = ScoresTeam1[7].Value;
                            if (ScoresTeam1[8])
                                s9t1 = ScoresTeam1[8].Value;
                            for (var st1 = 0; st1 < ScoresTeam1.length; st1++) {
                                var AwayScore = $('_Div_Score_' + leagueId + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_" + ScoresTeam1[st1].Id);
                                if (AwayScore && ScoresTeam1[st1].Value)
                                    AwayScore.update(ScoresTeam1[st1].Value);
                            }
                        }
                        if (ScoresTeam2) {
                            if (ScoresTeam2[0])
                                s1t2 = ScoresTeam2[0].Value;
                            if (ScoresTeam2[1])
                                s2t2 = ScoresTeam2[1].Value;
                            if (ScoresTeam2[2])
                                s3t2 = ScoresTeam2[2].Value;
                            if (ScoresTeam2[3])
                                s4t2 = ScoresTeam2[3].Value;
                            if (ScoresTeam2[4])
                                s5t2 = ScoresTeam2[4].Value;
                            if (ScoresTeam2[5])
                                s6t2 = ScoresTeam2[5].Value;
                            if (ScoresTeam2[6])
                                s7t2 = ScoresTeam2[6].Value;
                            if (ScoresTeam2[7])
                                s8t2 = ScoresTeam2[7].Value;
                            if (ScoresTeam2[8])
                                s9t2 = ScoresTeam2[8].Value;
                            for (var st2 = 0; st2 < ScoresTeam2.length; st2++) {
                                var HomeScore = $('_Div_Score_' + leagueId + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_" + ScoresTeam2[st2].Id);
                                if (HomeScore && ScoresTeam2[st2].Value)
                                    HomeScore.update(ScoresTeam2[st2].Value);
                            }
                        }
                        switch (leagueId) {
                        case"5":
                            {
                                var fiveScores1 = $('_Div_Score_' + leagueId + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_1st5");
                                if (fiveScores1) {
                                    if (s1t1 && s2t1 && s3t1 && s4t1 && s5t1)
                                        fiveScores1.update(parseInt(s1t1) + parseInt(s2t1) + parseInt(s3t1) + parseInt(s4t1) + parseInt(s5t1));
                                }
                                var fiveScores2 = $('_Div_Score_' + leagueId + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_1st5");
                                if (fiveScores2) {
                                    if (s1t2 && s2t2 && s3t2 && s4t2 && s5t2)
                                        fiveScores2.update(parseInt(s1t2) + parseInt(s2t2) + parseInt(s3t2) + parseInt(s4t2) + parseInt(s5t2));
                                }
                                break;
                            }
                        case"6":
                            {
                                var fiveScores11 = $('_Div_Score_' + leagueId + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_1st");
                                if (fiveScores11) {
                                    if (s1t1)
                                        fiveScores11.update(parseInt(s1t1));
                                }
                                var fiveScores21 = $('_Div_Score_' + leagueId + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_1st");
                                if (fiveScores21) {
                                    if (s1t2)
                                        fiveScores21.update(parseInt(s1t2));
                                }
                                var fiveScores12 = $('_Div_Score_' + leagueId + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_2nd");
                                if (fiveScores12) {
                                    if (s2t1)
                                        fiveScores12.update(parseInt(s2t1));
                                }
                                var fiveScores22 = $('_Div_Score_' + leagueId + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_2nd");
                                if (fiveScores22) {
                                    if (s2t2)
                                        fiveScores22.update(parseInt(s2t2));
                                }
                                break;
                            }
                        case"7":
                        case"2":
                        case"3":
                            {
                                var fiveScores11 = $('_Div_Score_' + leagueId + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_1st");
                                if (fiveScores11) {
                                    if (s1t1 && s2t1)
                                        fiveScores11.update(parseInt(s1t1) + parseInt(s2t1));
                                }
                                var fiveScores21 = $('_Div_Score_' + leagueId + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_1st");
                                if (fiveScores21) {
                                    if (s1t2 && s2t2)
                                        fiveScores21.update(parseInt(s1t2) + parseInt(s2t2));
                                }
                                var fiveScores12 = $('_Div_Score_' + leagueId + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_2nd");
                                if (fiveScores12) {
                                    if (s3t1 && s4t1)
                                        fiveScores12.update(parseInt(s3t1) + parseInt(s4t1));
                                }
                                var fiveScores22 = $('_Div_Score_' + leagueId + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_2nd");
                                if (fiveScores22) {
                                    if (s3t2 && s4t2)
                                        fiveScores22.update(parseInt(s3t2) + parseInt(s4t2));
                                }
                                break;
                            }
                        }
                        var FinalScore1 = $('_Div_Score_' + leagueId + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_final");
                        var FinalScore2 = $('_Div_Score_' + leagueId + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_final");
                        if (FinalScore1 && FinalScore2) {
                            if ((listEvents[i + 2].status != '' || listEvents[i + 2].status != '0') && (listEvents[i + 2].period != '' || listEvents[i + 2].period != '0')) {
                                if (listEvents[i + 2].awayScore)
                                    FinalScore1.update(listEvents[i + 2].awayScore);
                                if (listEvents[i + 2].homeScore)
                                    FinalScore2.update(listEvents[i + 2].homeScore);
                            }
                        }
                        var divLine1 = $('_Div_Line_' + leagueId + "_" + listEvents[i].EventId + "_" + listEvents[i].TeamId + "_" + listEvents[i].BookOpenerId);
                        var divLine2 = $('_Div_Line_' + leagueId + "_" + listEvents[i + 1].EventId + "_" + listEvents[i + 1].TeamId + "_" + listEvents[i + 1].BookOpenerId);
                        if (divLine1 != null && divLine2 != null) {
                            var BooksTeam1 = listEvents[i].Books;
                            var BooksTeam2 = listEvents[i + 1].Books;
                            var moneyTeam1;
                            var moneyTeam2;
                            var spreadTeam1;
                            var spreadTeam2;
                            var totalTeam1;
                            var totalTeam2;
                            for (var j = 0; j < BooksTeam1.length; j++) {
                                if (BooksTeam1[j].BookId == listEvents[i].BookOpenerId) {
                                    moneyTeam1 = BooksTeam1[j].Money;
                                    spreadTeam1 = BooksTeam1[j].Spread;
                                    totalTeam1 = BooksTeam1[j].Total;
                                }
                            }
                            for (var z = 0; z < BooksTeam2.length; z++) {
                                if (BooksTeam2[z].BookId == listEvents[i].BookOpenerId) {
                                    moneyTeam2 = BooksTeam2[z].Money;
                                    spreadTeam2 = BooksTeam2[z].Spread;
                                    totalTeam2 = BooksTeam2[z].Total;
                                }
                            }
                            if (lineAtt == "Money") {
                                if (moneyTeam1 && moneyTeam2 && moneyTeam1 < moneyTeam2) {
                                    if (moneyTeam1 == "0") {
                                        divLine1.update("PK");
                                    } else {
                                        divLine1.update(moneyTeam1);
                                    }
                                    if (totalTeam2)
                                        divLine2.update(totalTeam2);
                                } else {
                                    if (moneyTeam1 && moneyTeam2) {
                                        if (moneyTeam2 == "0") {
                                            divLine2.update("PK");
                                        } else {
                                            divLine2.update(moneyTeam2);
                                        }
                                        if (totalTeam1)
                                            divLine1.update(totalTeam1);
                                    }
                                }
                            } else {
                                if (spreadTeam1 && spreadTeam2 && spreadTeam1 < spreadTeam2) {
                                    if (spreadTeam1 == "0") {
                                        divLine1.update("PK");
                                    } else {
                                        divLine1.update(spreadTeam1);
                                    }
                                    if (totalTeam2)
                                        divLine2.update(totalTeam2);
                                } else {
                                    if (spreadTeam1 && spreadTeam2) {
                                        if (spreadTeam2 == "0") {
                                            divLine2.update("PK");
                                        } else {
                                            divLine2.update(spreadTeam2);
                                        }
                                        if (totalTeam1)
                                            divLine1.update(totalTeam1);
                                    }
                                }
                            }
                        }
                        var divStatus = $("_Div_Status_" + leagueId + "_" + listEvents[i].EventId);
                        var divPeriod = $("_Div_Period_" + leagueId + "_" + listEvents[i].EventId);
                        if (divStatus != null && divPeriod != null) {
                            if (listEvents[i + 2].period == "Final") {
                                divStatus.update("&nbsp");
                                divPeriod.update("FINAL");
                            } else {
                                if (listEvents[i + 2].status && listEvents[i + 2].status != "0")
                                    divStatus.update(listEvents[i + 2].status);
                                if (listEvents[i + 2].period && listEvents[i + 2].period != "0")
                                    divPeriod.update(listEvents[i + 2].period);
                            }
                        }
                    }
                }
            }.bind(this),
            onComplete: function(t) {
                this.SendingAjax = false;
            }.bind(this)
        });
    },
    FractionIt: function(number) {
        var ret = number.replace(".5", "&frac12;").replace(".0", "");
        if (ret != null || ret != "") {
            return ret;
        } else {
            return "-";
        }
    },
    BoldWinner: function(s1, s2, value) {
        if (parseInt(s1) > parseInt(s2))
            return "<span class='b'>" + value + "</span>";
        return value;
    },
    ConvertTime: function(div, time) {
        new Ajax.Request('/ajax/?a=[DonbestCommon]Statistic_ConvertTime', {
            method: 'post',
            parameters: {
                time: time
            },
            onSuccess: function(t) {
                var timeConverted = t.responseJSON.time;
                div.update(timeConverted);
            }.bind(this),
            onComplete: function(t) {}.bind(this)
        });
        return timeConverted;
    },
    getLinesCookie: function(cookieName) {
        var theCookie = "" + document.cookie;
        var ind = theCookie.indexOf(cookieName);
        if (ind==-1 || cookieName == "")
            return "";
        var ind1 = theCookie.indexOf(';', ind);
        if (ind1==-1)
            ind1 = theCookie.length;
        return unescape(theCookie.substring(ind + cookieName.length + 1, ind1));
    },
    callUpdateLinesMini: function() {
        this.firstTime = false;
        var ts = this.getLinesCookie('linesTimeStamp');
        var leagueId = this.options.LeagueId;
        this.UpdateLines(leagueId, this.options.timestamp, this.options.date);
    },
    refreshLinesMini: function(obj) {
        if (this.firstTime)
            Event.observe(window, 'load', function() {
                this.callUpdateLinesMini();
            }.bind(this));
        setInterval(obj + ".callUpdateLinesMini()", this.TIME_REFRESH);
    },
    callUpdateLinesOdds: function() {
        var ts = this.getLinesCookie('linesTimeStamp');
        var leagueId = this.options.LeagueId;
        this.UpdateLinesByType(leagueId, this.options.timestamp, this.options.date);
    },
    refreshLinesOdds: function(obj) {
        setInterval(obj + ".callUpdateLinesOdds()", this.TIME_REFRESH);
    },
    callUpdateLinesScores: function() {
        var ts = this.getLinesCookie('linesTimeStamp');
        var leagueId = this.options.LeagueId;
        this.UpdateLinesScores(leagueId, this.options.timestamp, this.options.date);
    },
    refreshLinesScores: function(obj) {
        setInterval(obj + ".callUpdateLinesScores()", this.TIME_REFRESH);
    },
    callUpdateLinesAllScores: function() {
        var ts = this.getLinesCookie('linesTimeStamp');
        var leagueId = this.options.LeagueId;
        this.UpdateLinesAllScores(leagueId, this.options.timestamp, this.options.date);
    },
    refreshLinesAllScores: function(obj) {
        setInterval(obj + ".callUpdateLinesAllScores()", this.TIME_REFRESH);
    },
    callUpdateLinesAllOdds: function() {
        var ts = this.getLinesCookie('linesTimeStamp');
        var leagueId = this.options.LeagueId;
        this.UpdateLinesAllOdds(leagueId, this.options.timestamp, this.options.date);
    },
    refreshLinesAllOdds: function(obj) {
        setInterval(obj + ".callUpdateLinesAllOdds()", this.TIME_REFRESH);
    }
});
var subMenu = {
    _div: null,
    MaxWidth: 910,
    MovedItems: function() {
        if (subMenu._div == null) {
            subMenu._div = new Element('div');
        }
        return subMenu._div;
    },
    ShowTo: 0,
    NewMenuHolder: null,
    MainHolder: null,
    MoreLi: null,
    Make: function(holder) {
        subMenu.MainHolder = holder;
        if (holder == null)
            return;
        var currentWidth = 0;
        currentWidth += parseInt(holder.down('ul').getStyle('margin-left'));
        var items = new Array();
        $(holder).select('li').each(function(li, i) {
            currentWidth += $(li).getWidth();
            if (currentWidth > subMenu.MaxWidth) {
                li.hide();
                subMenu.MoveItem(new Element('li').update(li.innerHTML));
            }
        });
        if (subMenu.MovedItems().select('li').length > 0) {
            subMenu.AddMoreButton();
            Event.observe(window, 'load', function() {
                subMenu.MakeMenu();
            });
        }
    },
    MoveItem: function(li) {
        subMenu.MovedItems().appendChild(li);
    },
    AddMoreButton: function() {
        var key = 'sectionMenuMoreHolder';
        subMenu.MoreLi = new Element('li', {
            'class': key + '_more',
            'id': key + '_more'
        }).update('More <img src="/donbest/images/db/more_dd.gif" style="vertical-align: bottom;" />');
        subMenu.MoreLi.className = key + '_more';
        subMenu.MainHolder.down('ul').insertBefore(subMenu.MoreLi, subMenu.MainHolder.down('ul').firstChild);
    },
    MakeMenu: function() {
        var key = 'sectionMenuMoreHolder';
        subMenu.NewMenuHolder = new Element('div', {
            'class': key,
            'id': key
        }).setStyle({
            display: 'none'
        });
        subMenu.NewMenuHolder.className = key;
        subMenu.NewMenuHolder.appendChild(new Element('ul'));
        var ul = subMenu.NewMenuHolder.down('ul');
        subMenu.MovedItems().select('li').each(function(li, i) {
            ul.appendChild(li);
        });
        $(document.body).appendChild(subMenu.NewMenuHolder);
        Event.observe(subMenu.MoreLi, 'mouseover', function() {
            subMenu.Show();
        });
        Event.observe(subMenu.MoreLi, 'mouseout', function() {
            subMenu.Hide();
        });
        Event.observe(subMenu.NewMenuHolder, 'mouseover', function() {
            subMenu.Show();
        });
        Event.observe(subMenu.NewMenuHolder, 'mouseout', function() {
            subMenu.Hide();
        });
    },
    Show: function() {
        clearTimeout(subMenu.ShowTo);
        if (subMenu.NewMenuHolder.visible())
            return;
        var pos = subMenu.MoreLi.cumulativeOffset();
        subMenu.NewMenuHolder.setStyle({
            top: (pos.top + subMenu.MoreLi.getHeight()) + 'px',
            left: ((pos.left + subMenu.MoreLi.getWidth()) - subMenu.NewMenuHolder.getWidth()) + 'px'
        });
        subMenu.NewMenuHolder.show();
        subMenu.MoreLi.addClassName('sectionMenuMoreHolder_more_selected');
    },
    Hide: function() {
        subMenu.ShowTo = setTimeout(function() {
            subMenu.MoreLi.removeClassName('sectionMenuMoreHolder_more_selected');
            subMenu.NewMenuHolder.hide();
        }, 300);
    }
}
if (typeof deconcept == "undefined") {
    var deconcept = new Object();
}
if (typeof deconcept.util == "undefined") {
    deconcept.util = new Object();
}
if (typeof deconcept.SWFObjectUtil == "undefined") {
    deconcept.SWFObjectUtil = new Object();
}
deconcept.SWFObject = function(_1, id, w, h, _5, c, _7, _8, _9, _a) {
    if (!document.getElementById) {
        return;
    }
    this.DETECT_KEY = _a ? _a : "detectflash";
    this.skipDetect = deconcept.util.getRequestParameter(this.DETECT_KEY);
    this.params = new Object();
    this.variables = new Object();
    this.attributes = new Array();
    if (_1) {
        this.setAttribute("swf", _1);
    }
    if (id) {
        this.setAttribute("id", id);
    }
    if (w) {
        this.setAttribute("width", w);
    }
    if (h) {
        this.setAttribute("height", h);
    }
    if (_5) {
        this.setAttribute("version", new deconcept.PlayerVersion(_5.toString().split(".")));
    }
    this.installedVer = deconcept.SWFObjectUtil.getPlayerVersion();
    if (!window.opera && document.all && this.installedVer.major > 7) {
        deconcept.SWFObject.doPrepUnload = true;
    }
    if (c) {
        this.addParam("bgcolor", c);
    }
    var q = _7 ? _7: "high";
    this.addParam("quality", q);
    this.setAttribute("useExpressInstall", false);
    this.setAttribute("doExpressInstall", false);
    var _c = (_8) ? _8: window.location;
    this.setAttribute("xiRedirectUrl", _c);
    this.setAttribute("redirectUrl", "");
    if (_9) {
        this.setAttribute("redirectUrl", _9);
    }
};
deconcept.SWFObject.prototype = {
    useExpressInstall: function(_d) {
        this.xiSWFPath=!_d ? "expressinstall.swf" : _d;
        this.setAttribute("useExpressInstall", true);
    },
    setAttribute: function(_e, _f) {
        this.attributes[_e] = _f;
    },
    getAttribute: function(_10) {
        return this.attributes[_10];
    },
    addParam: function(_11, _12) {
        this.params[_11] = _12;
    },
    getParams: function() {
        return this.params;
    },
    addVariable: function(_13, _14) {
        this.variables[_13] = _14;
    },
    getVariable: function(_15) {
        return this.variables[_15];
    },
    getVariables: function() {
        return this.variables;
    },
    getVariablePairs: function() {
        var _16 = new Array();
        var key;
        var _18 = this.getVariables();
        for (key in _18) {
            _16[_16.length] = key + "=" + _18[key];
        }
        return _16;
    },
    getSWFHTML: function() {
        var _19 = "";
        if (navigator.plugins && navigator.mimeTypes && navigator.mimeTypes.length) {
            if (this.getAttribute("doExpressInstall")) {
                this.addVariable("MMplayerType", "PlugIn");
                this.setAttribute("swf", this.xiSWFPath);
            }
            _19 = "<embed type=\"application/x-shockwave-flash\" src=\"" + this.getAttribute("swf") + "\" width=\"" + this.getAttribute("width") + "\" height=\"" + this.getAttribute("height") + "\" style=\"" + this.getAttribute("style") + "\"";
            _19 += " id=\"" + this.getAttribute("id") + "\" name=\"" + this.getAttribute("id") + "\" ";
            var _1a = this.getParams();
            for (var key in _1a) {
                _19 += [key] + "=\"" + _1a[key] + "\" ";
            }
            var _1c = this.getVariablePairs().join("&");
            if (_1c.length > 0) {
                _19 += "flashvars=\"" + _1c + "\"";
            }
            _19 += "/>";
        } else {
            if (this.getAttribute("doExpressInstall")) {
                this.addVariable("MMplayerType", "ActiveX");
                this.setAttribute("swf", this.xiSWFPath);
            }
            _19 = "<object id=\"" + this.getAttribute("id") + "\" classid=\"clsid:D27CDB6E-AE6D-11cf-96B8-444553540000\" width=\"" + this.getAttribute("width") + "\" height=\"" + this.getAttribute("height") + "\" style=\"" + this.getAttribute("style") + "\">";
            _19 += "<param name=\"movie\" value=\"" + this.getAttribute("swf") + "\" />";
            var _1d = this.getParams();
            for (var key in _1d) {
                _19 += "<param name=\"" + key + "\" value=\"" + _1d[key] + "\" />";
            }
            var _1f = this.getVariablePairs().join("&");
            if (_1f.length > 0) {
                _19 += "<param name=\"flashvars\" value=\"" + _1f + "\" />";
            }
            _19 += "</object>";
        }
        return _19;
    },
    write: function(_20) {
        if (this.getAttribute("useExpressInstall")) {
            var _21 = new deconcept.PlayerVersion([6, 0, 65]);
            if (this.installedVer.versionIsValid(_21)&&!this.installedVer.versionIsValid(this.getAttribute("version"))) {
                this.setAttribute("doExpressInstall", true);
                this.addVariable("MMredirectURL", escape(this.getAttribute("xiRedirectUrl")));
                document.title = document.title.slice(0, 47) + " - Flash Player Installation";
                this.addVariable("MMdoctitle", document.title);
            }
        }
        if (this.skipDetect || this.getAttribute("doExpressInstall") || this.installedVer.versionIsValid(this.getAttribute("version"))) {
            var n = (typeof _20 == "string") ? document.getElementById(_20): _20;
            n.innerHTML = this.getSWFHTML();
            return true;
        } else {
            if (this.getAttribute("redirectUrl") != "") {
                document.location.replace(this.getAttribute("redirectUrl"));
            }
        }
        return false;
    }
};
deconcept.SWFObjectUtil.getPlayerVersion = function() {
    var _23 = new deconcept.PlayerVersion([0, 0, 0]);
    if (navigator.plugins && navigator.mimeTypes.length) {
        var x = navigator.plugins["Shockwave Flash"];
        if (x && x.description) {
            _23 = new deconcept.PlayerVersion(x.description.replace(/([a-zA-Z]|\s)+/, "").replace(/(\s+r|\s+b[0-9]+)/, ".").split("."));
        }
    } else {
        if (navigator.userAgent && navigator.userAgent.indexOf("Windows CE") >= 0) {
            var axo = 1;
            var _26 = 3;
            while (axo) {
                try {
                    _26++;
                    axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash." + _26);
                    _23 = new deconcept.PlayerVersion([_26, 0, 0]);
                } catch (e) {
                    axo = null;
                }
            }
        } else {
            try {
                var axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7");
            } catch (e) {
                try {
                    var axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6");
                    _23 = new deconcept.PlayerVersion([6, 0, 21]);
                    axo.AllowScriptAccess = "always";
                } catch (e) {
                    if (_23.major == 6) {
                        return _23;
                    }
                }
                try {
                    axo = new ActiveXObject("ShockwaveFlash.ShockwaveFlash");
                } catch (e) {}
            }
            if (axo != null) {
                _23 = new deconcept.PlayerVersion(axo.GetVariable("$version").split(" ")[1].split(","));
            }
        }
    }
    return _23;
};
deconcept.PlayerVersion = function(_29) {
    this.major = _29[0] != null ? parseInt(_29[0]) : 0;
    this.minor = _29[1] != null ? parseInt(_29[1]) : 0;
    this.rev = _29[2] != null ? parseInt(_29[2]) : 0;
};
deconcept.PlayerVersion.prototype.versionIsValid = function(fv) {
    if (this.major < fv.major) {
        return false;
    }
    if (this.major > fv.major) {
        return true;
    }
    if (this.minor < fv.minor) {
        return false;
    }
    if (this.minor > fv.minor) {
        return true;
    }
    if (this.rev < fv.rev) {
        return false;
    }
    return true;
};
deconcept.util = {
    getRequestParameter: function(_2b) {
        var q = document.location.search || document.location.hash;
        if (_2b == null) {
            return q;
        }
        if (q) {
            var _2d = q.substring(1).split("&");
            for (var i = 0; i < _2d.length; i++) {
                if (_2d[i].substring(0, _2d[i].indexOf("=")) == _2b) {
                    return _2d[i].substring((_2d[i].indexOf("=") + 1));
                }
            }
        }
        return "";
    }
};
deconcept.SWFObjectUtil.cleanupSWFs = function() {
    var _2f = document.getElementsByTagName("OBJECT");
    for (var i = _2f.length - 1; i >= 0; i--) {
        _2f[i].style.display = "none";
        for (var x in _2f[i]) {
            if (typeof _2f[i][x] == "function") {
                _2f[i][x] = function() {};
            }
        }
    }
};
if (deconcept.SWFObject.doPrepUnload) {
    if (!deconcept.unloadSet) {
        deconcept.SWFObjectUtil.prepUnload = function() {
            __flash_unloadHandler = function() {};
            __flash_savedUnloadHandler = function() {};
            window.attachEvent("onunload", deconcept.SWFObjectUtil.cleanupSWFs);
        };
        window.attachEvent("onbeforeunload", deconcept.SWFObjectUtil.prepUnload);
        deconcept.unloadSet = true;
    }
}
if (!document.getElementById && document.all) {
    document.getElementById = function(id) {
        return document.all[id];
    };
}
var getQueryParamValue = deconcept.util.getRequestParameter;
var FlashObject = deconcept.SWFObject;
var SWFObject = deconcept.SWFObject;
var Tabs = {
    SendingAjax: false,
    _Div_Load: '',
    ChangeTab: function(obj, xml, xslt)
    {
        if (this.SendingAjax)
            return;
        this.SendingAjax = true;
        obj = $(obj);
        var divLoadingBull = $("_Div_Loading_Statistics");
        divLoadingBull.style.visibility = "visible";
        divLoadingBull.style.display = "block";
        new Ajax.Request('/ajax/?a=[DonbestCommon]Statistic_ChangeTabTransform', {
            method: 'post',
            parameters: {
                xmlPath: xml,
                xsltPath: xslt
            },
            onSuccess: function(t) {
                var data = t.responseJSON;
                $(obj.parentNode).select('li').each(function(li) {
                    li.removeClassName('leagueTabSelected');
                });
                obj.addClassName('leagueTabSelected');
                $('_DivOutput').update(data.html);
            }.bind(this),
            onComplete: function(t) {
                this.SendingAjax = false;
                divLoadingBull.style.visibility = "hidden";
                divLoadingBull.style.display = "none";
            }.bind(this)
        });
    },
    SelectedTab: function()
    {
        $('btnEdit').disabled = false;
        $('btnDelete').disabled = false;
    },
    EditTab: function(obj, moduleId, uniqueId)
    {
        $('btnAdd').disabled = true;
        $('btnEdit').disabled = true;
        $('btnDelete').disabled = true;
        $('ddTabs').disabled = true;
        $('ddXmlUrl').disabled = false;
        $('ddXsltUrl').disabled = false;
        var editContainer = new Element('div', {
            'class': 'pictureTitle'
        });
        var tabTxt = new Element('input', {
            type: 'text',
            value: $('ddTabs').value
        });
        var okBtn = new Element('input', {
            type: 'button',
            value: 'Save',
            'class': 'button'
        });
        var cancelBtn = new Element('input', {
            type: 'button',
            value: 'Cancel',
            'class': 'button'
        });
        Event.observe(okBtn, 'click', function(e) {
            this.SaveEditTab(editContainer, tabTxt, $('ddTabs').value);
        }.bindAsEventListener(this));
        Event.observe(cancelBtn, 'click', function(e) {
            this.HideTitlePopUp(editContainer);
        }.bindAsEventListener(this));
        editContainer.appendChild(tabTxt);
        editContainer.appendChild(okBtn);
        editContainer.appendChild(cancelBtn);
        obj.parentNode.appendChild(editContainer);
        var stringFiles = this.FindTab($('ddTabs').value);
        var xmlXslt = stringFiles.split(',');
        if (stringFiles != "" && xmlXslt.length != 0)
        {
            var xmlId = xmlXslt[0];
            var xsltId = xmlXslt[1];
            $("ddXmlUrl").value = xmlId;
            var xmlObj = new Object;
            xmlObj.value = xmlId;
            if (xmlId != "" && xmlId != "-1")
            {
                Transform.ChangeXml(xmlObj, moduleId, uniqueId, function() {
                    $('ddXsltUrl').value = xsltId;
                });
            }
        }
    },
    FindTab: function(tab)
    {
        var result = "";
        var textXMLTabs = $("txtTabs").value;
        var xmlDoc = this.LoadXml(textXMLTabs);
        var elements = xmlDoc.getElementsByTagName("tab");
        for (var i = 0; i < elements.length; i++)
        {
            var atts = elements[i].attributes;
            var nameAtt = atts.getNamedItem("name").nodeValue;
            if (nameAtt == tab)
            {
                result += atts.getNamedItem("xml").nodeValue;
                result += ",";
                result += atts.getNamedItem("xslt").nodeValue;
            }
        }
        return result;
    },
    AddTab: function(obj)
    {
        $('btnAdd').disabled = true;
        $('btnDelete').disabled = true;
        $('btnEdit').disabled = true;
        $('ddTabs').disabled = true;
        $('ddTabs').value = "";
        $('ddXmlUrl').value = "-1";
        $('ddXsltUrl').innerHTML = "";
        $('ddXmlUrl').disabled = false;
        $('ddXsltUrl').disabled = false;
        var addContainer = new Element('div', {
            'class': 'pictureTitle'
        });
        var tabTxt = new Element('input', {
            type: 'text'
        });
        var okBtn = new Element('input', {
            type: 'button',
            value: 'OK',
            'class': 'button'
        });
        var cancelBtn = new Element('input', {
            type: 'button',
            value: 'Cancel',
            'class': 'button'
        });
        Event.observe(okBtn, 'click', function(e) {
            this.AddTabToList(addContainer, tabTxt);
        }.bindAsEventListener(this));
        Event.observe(cancelBtn, 'click', function(e) {
            this.HideTitlePopUp(addContainer);
        }.bindAsEventListener(this));
        addContainer.appendChild(tabTxt);
        addContainer.appendChild(okBtn);
        addContainer.appendChild(cancelBtn);
        obj.parentNode.appendChild(addContainer);
    },
    HideTitlePopUp: function(addContainer)
    {
        $('btnAdd').disabled = false;
        $('ddTabs').value = "";
        $('ddXmlUrl').value = "-1";
        $('ddXsltUrl').innerHTML = "";
        $('ddXmlUrl').disabled = true;
        $('ddXsltUrl').disabled = true;
        $('btnDelete').disabled = true;
        $('ddTabs').disabled = false;
        addContainer.remove();
    },
    AddTabToList: function(addContainer, tabTxt)
    {
        var existsTab = this.FindTab(tabTxt.value);
        if (existsTab == "")
        {
            this.AddTabToXml(tabTxt.value);
            var tabs = $('ddTabs');
            var tabsDefault = $('ddTypeDefault');
            var newTab = new Element('option', {
                value: tabTxt.value
            });
            newTab.text = tabTxt.value;
            var newTabDef = new Element('option', {
                value: tabTxt.value
            });
            newTabDef.text = tabTxt.value;
            tabs.appendChild(newTab);
            tabsDefault.appendChild(newTabDef);
            addContainer.remove();
            $('btnAdd').disabled = false;
            $('ddTabs').value = "";
            $('ddXmlUrl').value = "-1";
            $('ddXsltUrl').innerHTML = "";
            $('ddXmlUrl').disabled = true;
            $('ddXsltUrl').disabled = true;
            $('btnDelete').disabled = true;
            $('ddTabs').disabled = false;
        } else
        {
            alert('This tab already exists.');
        }
    },
    SaveEditTab: function(editContainer, tabTxt, oldTab)
    {
        this.SaveEditTabAux(tabTxt.value, oldTab);
        editContainer.remove();
        $('btnAdd').disabled = false;
        $('ddTabs').value = "";
        $('ddXmlUrl').value = "-1";
        $('ddXsltUrl').innerHTML = "";
        $('ddXmlUrl').disabled = true;
        $('ddXsltUrl').disabled = true;
        $('btnDelete').disabled = true;
        $('ddTabs').disabled = false;
    },
    AddTabToXml: function(tabSelected)
    {
        var textXMLTabs = $("txtTabs").value;
        if (textXMLTabs == "")
        {
            textXMLTabs = "<Tabs></Tabs>";
        }
        var xmlDoc = this.LoadXml(textXMLTabs);
        var newTab = xmlDoc.createElement("tab");
        var newattName = xmlDoc.createAttribute("name");
        newattName.nodeValue = tabSelected;
        var newattXml = xmlDoc.createAttribute("xml");
        newattXml.nodeValue = $('ddXmlUrl').value == "-1" ? "" : $('ddXmlUrl').value;
        var newattXslt = xmlDoc.createAttribute("xslt");
        newattXslt.nodeValue = $('ddXsltUrl').value == "-1" ? "" : $('ddXsltUrl').value;
        newTab.setAttributeNode(newattName);
        newTab.setAttributeNode(newattXml);
        newTab.setAttributeNode(newattXslt);
        var root = xmlDoc.getElementsByTagName("Tabs")[0];
        root.appendChild(newTab);
        var string = (new XMLSerializer()).serializeToString(root);
        $("txtTabs").value = string;
    },
    SaveEditTabAux: function(newTab, oldTab)
    {
        var textXMLTabs = $("txtTabs").value;
        var xmlDoc = this.LoadXml(textXMLTabs);
        var elements = xmlDoc.getElementsByTagName("tab");
        for (var i = 0; i < elements.length; i++)
        {
            var atts = elements[i].attributes;
            var nameAtt = atts.getNamedItem("name").nodeValue;
            if (nameAtt == oldTab)
            {
                atts.getNamedItem("name").nodeValue = newTab;
                atts.getNamedItem("xml").nodeValue = $('ddXmlUrl').value == "-1" ? "" : $('ddXmlUrl').value;
                atts.getNamedItem("xslt").nodeValue = $('ddXsltUrl').value == "-1" ? "" : $('ddXsltUrl').value;
                var opts = $('ddTabs').options;
                for (var i = 0; i < opts.length; i++)
                {
                    var opt = opts[i];
                    if (opt.value == oldTab)
                    {
                        opt.value = newTab;
                        opt.text = newTab;
                        opt.selected = false;
                    }
                }
                var optsDef = $('ddTypeDefault').options;
                for (var i = 0; i < optsDef.length; i++)
                {
                    var opt = optsDef[i];
                    if (opt.value == oldTab)
                    {
                        opt.value = newTab;
                        opt.text = newTab;
                    }
                }
            }
        }
        var string = (new XMLSerializer()).serializeToString(xmlDoc);
        $("txtTabs").value = string;
    },
    DeleteTab: function()
    {
        var tabSelected = $('ddTabs').value;
        var textXMLTabs = $("txtTabs").value;
        var xmlDoc = this.LoadXml(textXMLTabs);
        var elements = xmlDoc.getElementsByTagName("tab");
        var opts = $('ddTabs').options;
        for (var i = 0; i < elements.length; i++)
        {
            var atts = elements[i].attributes;
            var nameAtt = atts.getNamedItem("name").nodeValue;
            if (nameAtt == tabSelected)
            {
                xmlDoc.documentElement.removeChild(elements[i]);
            }
        }
        for (var i = 0; i < opts.length; i++)
        {
            var opt = opts[i];
            if (opt.value == tabSelected)
            {
                $('ddTabs').removeChild(opt);
                $('ddTabs').value = "";
                $('ddXmlUrl').value = "-1";
                $('ddXsltUrl').innerHTML = "";
                $('btnEdit').disabled = true;
                $('btnDelete').disabled = true;
            }
        }
        var optsDef = $('ddTypeDefault').options;
        for (var i = 0; i < optsDef.length; i++)
        {
            var opt = optsDef[i];
            if (opt.value == tabSelected)
            {
                $('ddTypeDefault').removeChild(opt);
            }
        }
        var string = (new XMLSerializer()).serializeToString(xmlDoc);
        $("txtTabs").value = string;
    },
    LoadXml: function(xmlString)
    {
        if (window.DOMParser)
        {
            parser = new DOMParser();
            xmlDoc = parser.parseFromString(xmlString, "text/xml");
        } else
        {
            xmlDoc = new ActiveXObject("Microsoft.XMLDOM");
            xmlDoc.async = "false";
            xmlDoc.loadXML(xmlString);
        }
        return xmlDoc;
    }
}
var Transform = {
    SendingAjax: false,
    ChangeXslt: function(obj, moduleId, id)
    {
        if (this.SendingAjax)
            return;
        if (obj.value == "-1")
        {
            var newParams = $("parametersXSLT");
            newParams.innerHTML = "";
            newParams.style.display = 'none';
        }
        this.SendingAjax = true;
        new Ajax.Request('/ajax/?a=[DonbestCommon]Transform_ChangeXslt', {
            method: 'post',
            parameters: {
                xsltSelected: obj.value
            },
            requestHeaders: {
                Accept: 'application/json'
            },
            onSuccess: function(t) {
                parameters = t.responseJSON.html;
                var m = $('moduleData' + id);
                var pa;
                m.select('div.moduleSettings').each(function(div) {
                    pa = $("parametersXSLT");
                    if (!pa)
                    {
                        var divParameters = new Element('div', {
                            'class': 'parametersXSLT',
                            'id': 'parametersXSLT'
                        });
                        var contentParameters = new Element('div', {
                            'class': 'contentParameters',
                            'id': 'contentParameters'
                        });
                        divParameters.innerHTML = parameters;
                        contentParameters.appendChild(divParameters);
                        div.appendChild(contentParameters);
                    } else
                    {
                        pa.innerHTML = parameters;
                    }
                    var newParams = $("parametersXSLT");
                    if (newParams.innerHTML == "")
                    {
                        newParams.style.display = 'none';
                    } else
                    {
                        newParams.style.display = '';
                    }
                });
            }.bind(this),
            onComplete: function(t) {
                this.SendingAjax = false;
            }.bind(this)
        });
    },
    ChangeXml: function(obj, moduleId, id, func)
    {
        if (this.SendingAjax)
            return;
        if (obj.value == "-1")
        {
            var ddXslts = $("ddXsltUrl");
            if (ddXslts)
            {
                ddXslts.innerHTML = "";
                var newParams = $("parametersXSLT");
                newParams.innerHTML = "";
                newParams.style.display = 'none';
            }
        }
        this.SendingAjax = true;
        new Ajax.Request('/ajax/?a=[DonbestCommon]Transform_ChangeXml', {
            method: 'post',
            parameters: {
                xmlSelected: obj.value
            },
            requestHeaders: {
                Accept: 'application/json'
            },
            onSuccess: function(t) {
                var options = new Array;
                options = t.responseJSON.Options;
                var m = $('moduleData' + id);
                m.select('div.moduleSettings').each(function(div) {
                    var ddXslts = $("ddXsltUrl");
                    if (ddXslts)
                    {
                        ddXslts.innerHTML = "";
                        var elOptNew = document.createElement('option');
                        elOptNew.selected = true;
                        try {
                            ddXslts.add(elOptNew, null);
                        } catch (ex) {
                            ddXslts.add(elOptNew);
                        }
                        for (var i = 0; i < options.length; i++)
                        {
                            elOptNew = document.createElement('option');
                            var opt = options[i];
                            elOptNew.text = opt.path;
                            elOptNew.value = opt.id;
                            try {
                                ddXslts.add(elOptNew, null);
                            } catch (ex) {
                                ddXslts.add(elOptNew);
                            }
                        }
                        var newParams = $("parametersXSLT");
                        newParams.innerHTML = "";
                        newParams.style.display = 'none';
                    }
                });
            }.bind(this),
            onComplete: function(t) {
                if (typeof(func) != "undefined")
                {
                    func();
                }
                this.SendingAjax = false;
            }.bind(this)
        });
    }
}
