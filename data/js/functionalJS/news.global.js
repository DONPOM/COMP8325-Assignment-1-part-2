Netdania = Netdania || {};
Netdania.News = Netdania.News || {};
Netdania.News.Global = Netdania.News.Global || {};

Netdania.News.Global.RenderHeadlines = function (config) {

    var host = Netdania.General.Config.Host; //the NetDania server to connect to
    var connectionConfig = {
        pollingInterval: 1000, //milliseconds
        host: host,
        behavior: Netdania.JsApi.ConnectionType.LONGPOLLING, //LONG POLLING
        type: 1 //the server response format: JSONP
    }
    //create a json connection object
    var _connection = new Netdania.JsApi.JSONConnection(connectionConfig);
    _connection.Connect();

    var _requestNewsStory = function (newsId, provider, source, containerId, maxLength) {

        var reqStory = Netdania.JsApi.Request.getReqObjStory(newsId, provider);
     
        _connection.AddRequest(reqStory);
        

        _connection.addListener(Netdania.JsApi.Events.ONNEWSSTORY, function (newsStoryResponse) {
            if (newsStoryResponse.id == reqStory.i) {
                var _storyContent = newsStoryResponse.data;
                if (_storyContent.indexOf('Quotes from') > -1) {
                    _storyContent = _storyContent.substring(_storyContent.indexOf('</p>') + 5, _storyContent.length);
                }
                $("#" + containerId).html(_storyContent);
                if ($("#" + containerId).text().length > maxLength) {
                    $("#" + containerId).text(_textAbstract($("#" + containerId).text(), maxLength));
                    $("#" + containerId).html('<p>' + $("#" + containerId).html() + '<span id="nd-readMoreDetails_' + containerId + '" class="homepage-global-readmore">read more</span></p>');
                }
                $('#nd-readMoreDetails_' + containerId).click(function () {
                    var newsGeneralConfig = new Netdania.News.General.Config();
                    var selectedObject = $("#" + containerId).parent().find('.nd-homepage-global-headline-h');
                    window.location.href = newsGeneralConfig.NewsStory + '?id=' + selectedObject.attr("i") + '&p=' +
                    selectedObject.attr("p") + '&s=' + selectedObject.attr("source") + '&t=' + selectedObject.attr("t") +
                    '&h=' + encodeURIComponent(selectedObject.text());
                });
                Netdania.Util.overlayMask($('#' + containerId), 'hide');
            }
        });
    }

    var instruments = config.records;
    for (var i = 0; i < instruments.length; i++) {
        var headline = instruments[i].data.h.replace('RESEARCH: ', '').replace('NetDania: ', '');
        var html = '<table width="100%" collspacing="0" collpadding="0" border="0" ><tr>' +
                            '<td width="110px;"><div class="nd-homepage-global-news"><img src="\'\'" id="nd-homepage-global-headline' + i + '-img" class="nd-homepage-global-headline-img"/></div></td>' +
                            '<td><div class="nd-homepage-global-news" id="nd-homepage-global-headline' + i + '-news">' +
                            '<div id="nd-homepage-global-headline' + i + '-h" class="nd-homepage-global-headline-h" i="' + instruments[i].data.i.replace(/"/g, '%22') + '" p="' + instruments[i].data.provider + '" source="' + instruments[i].data.source + '" t="' + instruments[i].data.t + '">' + headline + '</div><div id="nd-homepage-global-headline' + i + '-story" ></div></div></td>' +
                        '</tr></table>';
        $('#nd-homepage-global-headline' + i).html(html);

        var relatedInstrumentsJSON;
        try {
            relatedInstrumentsJSON = jQuery.parseJSON(instruments[i].data.i.substring(instruments[i].data.i.indexOf("{"), instruments[i].data.i.length));
        }
        catch (exception) {

        }
        $('#nd-homepage-global-headline' + i + '-img').attr("src", Netdania.News.Utilities.getHeadlineImage(instruments[i].data.source, relatedInstrumentsJSON));
        $('#nd-homepage-global-headline' + i + '-img').shadow({ type: 'sides', sides: 'vt-2' });
        var instrument = instruments[i].data;
        $('#nd-homepage-global-headline' + i + '-h').click(function () {
            var newsGeneralConfig = new Netdania.News.General.Config();
            var selectedObject = $(this);
            window.location.href = newsGeneralConfig.NewsStory + '?id=' + selectedObject.attr("i") + '&p=' +
                    selectedObject.attr("p") + '&s=' + selectedObject.attr("source") + '&t=' + selectedObject.attr("t") +
                    '&h=' + encodeURIComponent(selectedObject.text());
        });


        var maxLength = $('#nd-homepage-global-headline' + i + '-news').width() - $('#nd-homepage-global-headline' + i + '-h').text().length - 10;
        _requestNewsStory(instruments[i].data.i, instruments[i].data.provider, instruments[i].data.source, "nd-homepage-global-headline" + i + "-story", maxLength);
        Netdania.Util.overlayMask($("nd-homepage-global-headline" + i + "-story"));
    }

    _connection.Flush();

    var _textAbstract = function (text, length) {
        if (text == null) {
            return "";
        }
        if (text.length <= length) {
            return text;
        }
        text = text.substring(0, length);
        last = text.lastIndexOf(" ");
        text = text.substring(0, last);
        return text + "...";
    }
}

Netdania.News.Global.RenderNewsTable = function (tableConfig) {
    var headerClass, contentClass, contentOddClass, width, fields, instr;

    var fields = {
        t: { name: 'Time', value: '', width: '10%', hdTDcls: '', TDCls: 'nd-homepage-news-TD', cellCls: 'nd-homepage-newsTimeColumn', hdCellCls: '' },
        h: { name: 'Headline', value: '', width: '90%', hdTDcls: '', TDCls: 'nd-homepage-news-TD', cellCls: 'nd-homepage-newsHeadlineColumn', hdCellCls: '' }
    };

    var renderers = {

        h: { fn: function (grid, rec, cellid, val, oldval, fxy) {
            var t = rec.data.t;
            var dateUnf = new Date(t * 1000);
            t = '';
            val = val.replace('RESEARCH: ', '').replace('NetDania: ', '');
            if (val.length > tableConfig.headlineMaxLength) {
                val = val.substring(0, tableConfig.headlineMaxLength - 3).substring(0, val.lastIndexOf(' '));
                val = val + '...';
            }
            var storyCls = (rec.data.s == 1) ? 'class ="nd-news-headline-row"' : 'class ="nd-news-headline-row-nohover"';
            return t + '<span ' + storyCls + '>' + val + '</span>';
        }
        },
        t: { fn: function (grid, rec, cellid, val, oldval, fxy) {
            if (val === '' || val === "N/A" || val === undefined) {
                return "";
            }
            var dateUnf = new Date(val * 1000);

            return '<span style="padding-left:5px">' + Netdania.Util.dateFormat(dateUnf, "HH:MM").toLowerCase() + '</span>';
        }
        }
    };

    var instr = tableConfig.records;
    var tableWidth = 0;
    var headerHtml = '';
    var colsHtml = '';
    var lastField = '';
    var contentClass = tableConfig.contentClass;
    if (tableConfig.contentClass === undefined || tableConfig.contentClass === null || tableConfig.contentClass === '') {
        contentClass = "gridDefaultContentClass";
    }
    var contentOddClass = tableConfig.contentOddClass;
    if (tableConfig.contentOddClass === undefined || tableConfig.contentOddClass === null || tableConfig.contentOddClass === '') {
        contentOddClass = "gridDefaultContentOddClass";
    }

    if (tableConfig.width !== undefined && tableConfig.width !== null) {
        tableWidth = tableConfig.width;
    }
    for (var prop in fields) {
        lastField = prop;
    }
    var cols = 0;

    var forceHeight = '';

    var align = '';
    if (tableConfig.align === '') {
        align = "center";
    }
    else {
        align = tableConfig.align;
    }
    var jshtml = '';

    jshtml = '<table id="' + tableConfig.renderTo + '" width="100%" border="0" cellspacing="0" cellpadding="0" >';

    jshtml += '<tbody>';


    var noDataMessage = tableConfig.noDataMessage;
    if (noDataMessage === undefined) {
        noDataMessage = '';
    }
    if (instr.length == 0) {
        jshtml += '<tr><td colspan=' + cols + ' class="netd-grid-empty">' + noDataMessage + '</td></tr>';
    }

    for (var ii = 0; ii < instr.length; ii++) {

        var recid = instr[ii].id;


        var relatedInstrumentsJSON;
        try {
            relatedInstrumentsJSON = jQuery.parseJSON(instr[ii].data.i.substring(instr[ii].data.i.indexOf("{"), instr[ii].data.i.length));
        }
        catch (exception) {

        }
        var alias = Netdania.News.Utilities.getAliasBySource(instr[ii].data.source, relatedInstrumentsJSON);


        jshtml += '<tr p="' + instr[ii].data.provider + '" s="' + instr[ii].data.s + '" id="' + instr[ii].data.i.replace(/"/g, '%22') + '" source="' + instr[ii].data.source + '" alias="' + alias + '"  t="' + instr[ii].data.t + '"  openFQ="' + instr[ii].data.openFQ + '" class="' + contentClass + '">';

        for (var prop1 in fields) {
            if (fields.hasOwnProperty(prop1)) {
                var cellid = recid + '-' + prop1;
                var originalValue = instr[ii].data[prop1];
                var value;
                if (originalValue === undefined) {
                    value = '';
                }
                if (renderers[prop1] !== undefined) {
                    value = renderers[prop1].fn(tableConfig, instr[ii], cellid, originalValue, value, prop1);
                } else {
                    value = '<div>' + originalValue + '</div>';
                }
                var lc = '';
                if (lastField === prop1) {
                    lc = ' lastColCls';
                }

                jshtml += '<td width="' + fields[prop1].width + '" wrap style="overflow: hidden;" class="' + fields[prop1].TDCls + lc + '"><div id="' + cellid + '" class="' + fields[prop1].cellCls + '" originalValue="' + originalValue + '" >' + value + '</div></td>';
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
        var el = document.getElementById(tableConfig.renderTo);
        el.innerHTML = jshtml;
        var _newsTable = $('#' + tableConfig.renderTo);
        var newsConfig = new Netdania.News.General.Config();
        $('#' + tableConfig.renderTo + ' tr').click(function () {
            if ($(this).attr("openFQ") !== '' && $(this).attr("openFQ") !== 'undefined') {
                id = decodeURIComponent($(this).attr("id"));
                var relatedInstrumentsJSON;
                try {
                    relatedInstrumentsJSON = jQuery.parseJSON(id.substring(id.indexOf("{"), id.length));
                }
                catch (exception) {

                }
                if (relatedInstrumentsJSON) {
                    if (relatedInstrumentsJSON.Syms.length == 1)
                        window.location.href = newsConfig.FullQuote + '?symbol=' + relatedInstrumentsJSON.Syms[0].sym + ((relatedInstrumentsJSON.Syms[0].prv !== undefined) ? '&provider=' + relatedInstrumentsJSON.Syms[0].prv : '');
                    return;
                }
            }
            
            window.location.href = newsConfig.NewsStory + '?id=' + $(this).attr("id") + '&p=' + $(this).attr("p") + '&s=' + $(this).attr("source") + '&t=' + $(this).attr("t") + '&h=' + encodeURIComponent($($(this).find('.nd-homepage-newsHeadlineColumn')).attr('originalValue'));

        });

        jshtml = null;
    }
    else {
        return jshtml;
    }
};
