
(function(window) {

    Array.prototype.indexOf = Array.prototype.indexOf || function(e) {
        for (var i = 0; i < this.length; i++) {
            if (this[i] == e) 
                return i;
        }
        return - 1;
    };

    var models = [{
        "UserId": 192984,
        "Name": "AvyAngelina",
        "Topic": "Hey I am Angy! If u want nudity go private!",
        "Age": 23,
        "Camscore": "32366",
        "Status": "Public",
        "ImageUrl": "http://content3.pluginnetwork.net/static/avatar_192984_av_0_180x135_10176_nohash.jpg",
        "IsBroadcasting": true,
        "LivePreviewHash": "192984:116bb562795d492ac81f233486fb8b02",
        "Url": "http://aff.camplace.com/delivery/?w=783&b=AU&c=DP&t=2&l=en&version=v2&redirectUrl=http%3A%2F%2Fwww.camplace.com%2Fwebcam-chat%2Favyangelina-192984.aspx",
        "Index": 0
    }, {
        "UserId": 287107,
        "Name": "IreneDoll",
        "Topic": "40tok Flash Boobs, 80tok Strip Show, 200tok Skype show!",
        "Age": 19,
        "Camscore": "39231",
        "Status": "Public",
        "ImageUrl": "http://content5.pluginnetwork.net/static/avatar_287107_av_0_180x135_10176_nohash.jpg",
        "IsBroadcasting": true,
        "LivePreviewHash": "287107:27374cfa7d8c76d87ce14fa33c6af451",
        "Url": "http://aff.camplace.com/delivery/?w=783&b=AU&c=DP&t=2&l=en&version=v2&redirectUrl=http%3A%2F%2Fwww.camplace.com%2Fwebcam-chat%2Firenedoll-287107.aspx",
        "Index": 1
    }, {
        "UserId": 28709,
        "Name": "KarlaAllure",
        "Topic": "Boobies 40 tokens , Pussy 60 tokens . KISS",
        "Age": 19,
        "Camscore": "45866",
        "Status": "Public",
        "ImageUrl": "http://content4.pluginnetwork.net/static/avatar_28709_av_0_180x135_10176_nohash.jpg",
        "IsBroadcasting": true,
        "LivePreviewHash": "28709:ba83eb07b6eca2eedd88678e54874d80",
        "Url": "http://aff.camplace.com/delivery/?w=783&b=AU&c=DP&t=2&l=en&version=v2&redirectUrl=http%3A%2F%2Fwww.camplace.com%2Fwebcam-chat%2Fkarlaallure-28709.aspx",
        "Index": 2
    }, {
        "UserId": 287560,
        "Name": "MeganS",
        "Topic": "30 tits 40watch cam  50 pussy 60finger pussy 70toys pussy 80finger ass 90toys ass 100strip 160squirt 200skype",
        "Age": 21,
        "Camscore": "41316",
        "Status": "Public",
        "ImageUrl": "http://content1.pluginnetwork.net/static/avatar_287560_av_0_180x135_10176_nohash.jpg",
        "IsBroadcasting": true,
        "LivePreviewHash": "287560:3232deed3824ff6f8c7c2f2897c39919",
        "Url": "http://aff.camplace.com/delivery/?w=783&b=AU&c=DP&t=2&l=en&version=v2&redirectUrl=http%3A%2F%2Fwww.camplace.com%2Fwebcam-chat%2Fmegans-287560.aspx",
        "Index": 3
    }, {
        "UserId": 147030,
        "Name": "BritnneyLee",
        "Topic": "Enjoy My Presence Here",
        "Age": 21,
        "Camscore": "37258",
        "Status": "Public",
        "ImageUrl": "http://content5.pluginnetwork.net/static/avatar_147030_av_0_180x135_10176_nohash.jpg",
        "IsBroadcasting": true,
        "LivePreviewHash": "147030:e0a96f438e1eb7d9f9d4a8691b8eb2f0",
        "Url": "http://aff.camplace.com/delivery/?w=783&b=AU&c=DP&t=2&l=en&version=v2&redirectUrl=http%3A%2F%2Fwww.camplace.com%2Fwebcam-chat%2Fbritnneylee-147030.aspx",
        "Index": 4
    }, {
        "UserId": 28029,
        "Name": "Krissy Belle",
        "Topic": "tip 11 if you like me, 20 if you want me, sensual and kinky shows in private",
        "Age": 23,
        "Camscore": "36198",
        "Status": "Public",
        "ImageUrl": "http://content2.pluginnetwork.net/static/avatar_28029_av_0_180x135_10176_nohash.jpg",
        "IsBroadcasting": true,
        "LivePreviewHash": "28029:97a8db73bc6535c347bc9f1fa4642c73",
        "Url": "http://aff.camplace.com/delivery/?w=783&b=AU&c=DP&t=2&l=en&version=v2&redirectUrl=http%3A%2F%2Fwww.camplace.com%2Fwebcam-chat%2Fkrissy-belle-28029.aspx",
        "Index": 5
    }
    ];

    var placeholders = ["livePreview", "Div1", "Div2", "Div3", "Div4", "Div5"];

    var all = false; //false; /*disabled for now*/
    var autorefresh = true;

    var autoStart = true;
    var renderMethod = null;
    var loadCallback = "showData";
    var muted = true;
    var canMute = true;
    var highQuality = true;

    // Other parameters

    var website = "http://www.camplace.com";

    var ignoreUserIds = [];

    if (!window.swfobject) {
        if (window.console && window.console.log)
            window.console.log("swfobject not found!");
        return;
    }

    if (!window.jQuery) {
        if (window.console && window.console.log)
            window.console.log("jQuery not found!");
        return;
    }

    window.Plugin = window.Plugin || {};

    window.Plugin.Models = models;

    var livePreviewId;

    window.StartLivePreview = function() {
        $("#" + livePreviewId).each(function() {
            $(this).get(0).Start();
        });
    }

    window.StopLivePreview = function() {
        $("#" + livePreviewId).each(function() {
            var p = $(this).get(0);
            // if (typeof p == "function")
            if (p && typeof p.Stop == "function") 
                p.Stop();
        });
    }

    var renderLivePreview = function (divId, model) {

        if (ignoreUserIds.indexOf(model.UserId) >= 0 ) {
            window.ResetModelPreview(model.Index);
            return;
        }

        ignoreUserIds.push(model.UserId);

        $("#" + divId).empty();

        var placeHolderId = "CamPreview" + String(Math.random()).substr(2);

        $("<div></div>").attr("id", placeHolderId).appendTo("#" + divId);

        if (model.Index == 0) {
            livePreviewId = placeHolderId;
        }

        var swfVersionStr = "11.4.0";
        var xiSwfUrlStr = "/flash/playerProductInstall.swf";

        var flashvars = {};
        flashvars.ResDir = website + "/flash/resources/"
        flashvars.Url = "rtmp://vidclients.pluginnetwork.com/PNPreview";
        flashvars.autoStart = autoStart;
        flashvars.videoHash = model.LivePreviewHash;
        flashvars.highQuality = highQuality;
        flashvars.muted = model.Index == 0 ? muted : "true";
        flashvars.canMute = canMute;
        //flashvars.loadCallback = loadCallback;
        flashvars.pictureUrl = model.ImageUrl;
        flashvars.modelStatus = model.Status;

        flashvars.redirectUrl = encodeURIComponent(model.Url);

        flashvars.modelName = model.Name;

        var params = {};
        params.scale = "noScale";
        params.quality = "high";
        params.wmode = "direct";
        params.bgcolor = "#ffffff";
        params.allowscriptaccess = "always";
        params.allowfullscreen = "true";
        params.wmode = "transparent";

        var attributes = {};
        attributes.id = placeHolderId;
        attributes.name = placeHolderId;
        //attributes.align = "middle";

        window.swfobject.embedSWF(website + "/flash/OtherVideoSample/OtherVideoSampleFlash.swf?rndm=" + Math.random(), placeHolderId,
        "100%", "100%",
        swfVersionStr, xiSwfUrlStr,
        flashvars, params, attributes);

        if (window.renderedLivePreview)
            window.renderedLivePreview(divId, model);
    }

    var renderThumbs = function(startIndex) {
        for (var index = startIndex; index < models.length; index ++) {
            var imageUrl = models[index].ImageUrl; //website + "/handle/image.ashx?userid=" + models[index].UserId + "&typecode=av&imageindex=0&size=Preview&fileextension=.jpg";
            $("#" + placeholders[index]).append(
            $("<a></a>").attr("href", models[index].Url)
            .attr("target", "_blank")
            .append(
            $("<img/>").attr("src", imageUrl)
            .css({
                width: "100%",
                height: "100%",
                order: 0 
            })
            )
            );
            if (window.renderedLivePreview)
                window.renderedLivePreview(placeholders[index], models[index]);
        }
    }

    if (placeholders.length) {
        jQuery(document).ready(function () {
            if (all) {
                for (var i = 0; i < placeholders.length; i++) {
                    renderLivePreview(placeholders[i], models[i]);
                }
            } else {
                renderThumbs(1);
                renderLivePreview(placeholders[0], models[0]);
            }
            if (loadCallback) {
                window[loadCallback](models);
            }
        });
    } else {
        window[renderMethod || "renderLivePreview"] = renderLivePreview;
    }

    window.FLSHOut = window.FLSHOut || function(msg) {
        if (window.console && window.console.log)
            window.console.log(msg);
    }

    window.ResetModelPreview = function(index) {

        var model = models[index];

        var callback = "callback" + String(Math.random()).substr(2);

        window[callback] = function(newModels) {
            var newModel = newModels[0];
            if (newModel) {
                if (newModel.Status == "Public") {
                    models[newModel.Index = model.Index] = newModel;
                    renderLivePreview(placeholders[newModel.Index], newModel);
                }
            }
        }

        var script = document.createElement("script");
        script.type = "text/javascript";
        script.src = website + "/handle/livepreview.ashx?jsonPCallback=" + callback + "&IgnoreUserIds=" + ignoreUserIds.join(",");
        document.body.appendChild(script);

    }

    window.LivePreviewNotify = function(hash, event) {

        if (window.console && window.console.log)
            window.console.log("LivePreviewNotify " + event);

        var userId = hash.split(":")[0];

        if (autorefresh && userId) {

            var model = models.filter(function(model) {
                return model.UserId == userId 
            })[0];

            if (model) {
                window.ResetModelPreview(model.Index);
            }
        }
    }
})(window)
