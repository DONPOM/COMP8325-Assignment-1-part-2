define(["jquery/nyt", "underscore/nyt", "backbone/nyt", "http://js.nyt.com/js2/build/video/libs/brightcove/BrightcoveExperiences_All.js", "videoFactory", "foundation/base-mixin", "shared/sharetools/views/sharetools"], function(e, t, n, r, i, s, o) {
    var u = {};
    return function(n) {
        n.Extended = function(t) {
            this.options = e.extend({
                name: "",
                token: "cE97ArV7TzqBzkmeRVVhJ8O6GWME2iG_bRvjBTlNb4o.",
                playerId: "3161148879001",
                publisherId: "1749339200",
                thumbWidth: 150,
                thumbsVisible: 10,
                thumbsDistance: 6,
                thumbsMax: null,
                videoWidth: 768,
                videoHeight: 432,
                playerType: "homepage",
                moreVideos: "http://www.nytimes.com/video",
                debug: !1
            }, t), this.element = this.options.container, this.video = null, this.rendered=!1, this.currentVideo = 1, this.preInactivityTimerEnableDuration = 1e3, this.$videoPlaylist = null, this.$poster = null, this.$videoPlayer = null, this.count = 10, this.skip = 0, this.videoPlaylist = this.options.videoPlaylist, this.requestData=!0, this.visibleTimer = null, this.thumbOuterHeight = 0, this.initialize()
        }, n.Extended.prototype = {
            initialize: function() {
                var e = this;
                this.visibleTimer = setTimeout(function() {
                    e.unHideVideoRegion()
                }, 3e3), this.getData(this.count, this.skip, function(t) {
                    e.playlist = t, e.setVideoIds(), e.render(), e.registerEvents(), e.setMoreVideosLink(), e.skip += 10, e.unHideVideoRegion(), s.broadcast("nyt:video-playlist-ready"), e.options.onReady && typeof e.options.onReady == "function" && e.options.onReady.call(e)
                })
            },
            unHideVideoRegion: function() {
                e(".video-player-region").removeClass("invisible")
            },
            getStaticDomain: function() {
                var e = window.location.host, t = "http://js.nyt.com/";
                if (this.options.debug)
                    return "http://static.local.nytimes.com/";
                switch (e) {
                case"localhost":
                    t = "http://static.local.nytimes.com/";
                    break;
                case"video-int01.qprvt.nytimes.com":
                    t = "http://video-int01.qprvt.nytimes.com/video/";
                    break;
                case"www.dev.nytimes.com":
                    t = "http://static.dev.nytimes.com/";
                    break;
                case"www.stg.nytimes.com":
                    t = "http://static.stg.nytimes.com/";
                    break;
                default:
                    t = "http://js.nyt.com/"
                }
                return t
            },
            getGraphicsDomain: function() {
                var e = window.location.host, t = "http://graphics8.nytimes.com/";
                if (this.options.debug)
                    return "http://static.local.nytimes.com/";
                switch (e) {
                case"localhost":
                    t = "http://static.local.nytimes.com/";
                    break;
                case"video-int01.qprvt.nytimes.com":
                    t = "http://video-int01.qprvt.nytimes.com/video/";
                    break;
                case"www.dev.nytimes.com":
                    t = "http://static.dev.nytimes.com/";
                    break;
                case"www.stg.nytimes.com":
                    t = "http://static.stg.nytimes.com/";
                    break;
                default:
                    t = "http://graphics8.nytimes.com/"
                }
                return t
            },
            videoIds: [],
            registerEvents: function() {
                var t = this, n = ".video-playlist li", r = ".video-player .media-action-overlay";
                this.element.on("click", n, function() {
                    var n = e(this);
                    t.selectThumb(n)
                }), this.element.on("click", r, function() {
                    t.renderVideo()
                })
            },
            getDataHost: function() {
                var e = window.location.host, t;
                return e.indexOf("stg")>-1 || e.indexOf("sbx")>-1 ? t = "http://www.stg.nytimes.com/" : t = "http://www.nytimes.com/", t
            },
            getData: function(t, n, r) {
                var i = this.getDataHost() + "svc/video/api/v2/playlist/" + this.options.referenceId + "?count=" + t + "&skip=" + n;
                e.ajax({
                    url: i,
                    async: !1,
                    type: "get",
                    dataType: "jsonp",
                    cache: !0,
                    success: r,
                    jsonpCallback: "getPlayistCallback"
                })
            },
            selectThumb: function(e) {
                var t = e.index(), n = this.playlist.videos[t];
                this.$videoPlaylist.find("li").removeClass("selected"), e.addClass("selected"), this.currentVideo = t, this.renderVideoMetaData(n), this.renderVideo(n), this.updateShareTool(n), this.scrollTo(e)
            },
            setVideoIds: function() {
                var e = this.playlist.videos, t = this.options.thumbsMax || e.length;
                for (var n = 0; n < t; n++)
                    this.videoIds.push(e[n].id)
            },
            getKicker: function(e) {
                var t = /nyto_([^,]+)/.exec(e.join(","))[1];
                return t || ""
            },
            render: function() {
                this.setVariables(), this.renderThumbs(), this.renderPoster(), this.updateShareTool(), this.thumbTuckUnder()
            },
            setVariables: function() {
                this.$videoPlaylist = this.element.find(".video-playlist"), this.$poster = this.element.find(".poster"), this.$videoPlayer = this.element.find(".video-player")
            },
            renderPoster: function() {
                var e = this.playlist.videos[0], t = ["videoSixteenByNine768", "videoSmall"], n = this.getGraphicsDomain() + this.getCrop(0, this.playlist.videos, t, "poster"), r = this.$videoPlayer.find(".media-action-overlay");
                this.$poster.attr("src", n), this.renderVideoMetaData(e), e.playlist.parent_channel.headline !== undefined ? r.find(".kicker-label").html(e.playlist.parent_channel.headline) : r.find(".kicker-label").remove(), e.playlist.parent_channel.headline !== e.playlist.headline ? (r.find(".channel").html(e.playlist.headline), e.playlist.parent_channel.headline === undefined && r.find(".channel").css("margin-left", 0)) : r.find(".channel").remove(), r.find(".duration").html(this.formatDuration(e.duration)), r.find(".headline").html(e.headline), r.find(".summary").html(e.summary), this.element.removeClass("invisible")
            },
            renderVideoMetaData: function(e) {
                var t = "", n = this.element.find("figcaption .byline");
                e.byline && (t = "By " + e.byline + ". "), e.images[0] && e.images[0].credit && (t = t + "Image by " + e.images[0].credit + "."), n.html(t), this.element.find("figcaption .user-action").attr("href", [e.domain, e.publish_url].join(""))
            },
            renderVideo: function(e) {
                var t, n, r, o = this, u = e || this.playlist.videos[0];
                if (this.video && this.video.videoModule) {
                    this.video.videoModule.loadVideo(u.id, "referenceId");
                    return 
                }
                t=!0, n=!0, r = this.options.thumbsMax || this.playlist.videos.length;
                var a = this.element.find(".video-player-container");
                this.$poster.hide(), this.$videoPlayer.find(".media-action-overlay").hide(), a.attr("id", "video_" + u.id), a.empty(), o.video = i.create({
                    container: "video_" + u.id,
                    playerId: o.options.playerId,
                    publisherId: o.options.publisherId,
                    publisherReadToken: o.options.token,
                    videoId: u.id,
                    width: o.options.videoWidth,
                    height: o.options.videoHeight,
                    autoStart: n,
                    autoRender: t,
                    playerType: o.options.playerType,
                    userInitiatedPlay: !0,
                    siteSection: "homepage_large_homepage"
                });
                var f = o.video.getSubscription();
                f.subscribe(i.VideoEvents.RENDERED, function() {
                    var e = o.$videoPlaylist.find(".selected");
                    e.length === 0 && o.$videoPlaylist.find("li:first").addClass("selected"), o.rendered=!0, window.magnum.device === "mobile" && o.video.videoModule.play()
                }), f.subscribe(i.VideoEvents.VIDEO_COMPLETE, function() {
                    setTimeout(function() {
                        if (r > o.currentVideo + 1) {
                            var e = o.$videoPlaylist.find(".selected").next();
                            e.length === 0 && (e = o.$videoPlaylist.find("li:first").next()), e && o.selectThumb(e)
                        }
                    }, 100)
                }), f.subscribe(i.VideoEvents.MEDIA_BEGIN, function(e) {
                    s.broadcast("nyt:inactivity-timer-disable")
                }), f.subscribe(i.VideoEvents.VIDEO_START, function(e) {
                    s.broadcast("nyt:inactivity-timer-disable")
                }), f.subscribe(i.VideoEvents.VIDEO_STOP, function(e) {
                    setTimeout(function() {
                        s.broadcast("nyt:inactivity-timer-enable")
                    }, this.preInactivityTimerEnableDuration)
                }), f.subscribe(i.VideoEvents.VIDEO_PAUSED, function(e) {
                    s.broadcast("nyt:inactivity-timer-enable")
                }), f.subscribe(i.VideoEvents.AD_START, function(e) {
                    s.broadcast("nyt:inactivity-timer-disable")
                }), f.subscribe(i.VideoEvents.AD_COMPLETE, function(e) {
                    s.broadcast("nyt:inactivity-timer-enable")
                })
            },
            renderThumbs: function(t) {
                var n, r, i, s, o, u, a = this.playlist.videos, f = this, l = this.element.find(".loader-container"), c = ["videoSixteenByNine150", "wide"];
                t && (a = t.videos), l && this.handleLoader("remove"), s = this.options.thumbsMax || a.length, e(".video-playlist-container").scroll(function() {
                    f.scrollPercentage = 100 * this.scrollTop / (this.scrollHeight - this.clientHeight)
                }), this.updateScroll();
                for (o = 0; o < s; o++)
                    u = a[o].playlist.headline, a[o].playlist.parent_channel && a[o].playlist.parent_channel.headline && (u = a[o].playlist.parent_channel.headline), n = [this.getGraphicsDomain(), this.getCrop(o, a, c, "thumbs")].join(""), r = "", i = e('<li><article class="story"><div class="wide-thumb"><img src="' + n + '"></img>' + '<div class="media-action-overlay">' + '<div class="icon-container"><i class="icon"></i></div><p class="media-meta">Now Playing</p>' + "</div>" + "</div>" + '<h3 class="kicker"><span class="kicker-label">' + u + "</span><span class=duration>" + this.formatDuration(a[o].duration) + "</span></h3>" + '<h2 class="headline">' + a[o].headline + "</h2>" + "</article></li>"), this.$videoPlaylist.append(i);
                this.handleLoader("append")
            },
            thumbTuckUnder: function() {
                var e = 152, t = this.element.find(".video-playlist-container");
                t.animate({
                    scrollTop: e
                })
            },
            renderMoreButton: function() {
                var t = e('<li class ="button-container"><a href="http://www.nytimes.com/video?src=vidm" class="user-action button">More videos &raquo;</a></li>');
                this.$videoPlaylist.append(t), this.handleLoader("remove")
            },
            getCrop: function(e, t, n, r) {
                var i, s, o = t[e].images;
                for (i = 0; i < n.length; i++)
                    for (s = 0; s < o.length; s++)
                        if (o[s].type === n[i] && o[s].url !== "")
                            return o[s].url;
                return r === "poster" ? "video/static-timesvideo/img/video-default-16by9-1050x591.png" : "video/static-timesvideo/img/video-default-16by9-225x126.png"
            },
            updateScroll: function() {
                window.requestAnimationFrame(t.bind(this.updateScroll, this));
                var e = this, n = 1;
                e.scrollPercentage >= 99 && e.requestData && (e.requestData=!1, n += 1, e.getData(e.count, e.skip, function(t) {
                    e.renderThumbs(t), e.skip += 10, e.requestData=!0, e.playlist.videos = e.playlist.videos.concat(t.videos), e.playlist.videos.length === 100 && (e.requestData=!1, e.renderMoreButton())
                }))
            },
            handleLoader: function(t) {
                var n;
                t === "append" ? n = e('<li class="loader-container"><div class="loader"><span class="visually-hidden">Loading...</span></div></li>').appendTo(this.$videoPlaylist) : this.$videoPlaylist.find(".loader-container").remove()
            },
            scrollTo: function(e) {
                var t = e.position().top, n = this.element.find(".video-playlist-container");
                n.animate({
                    scrollTop: t
                })
            },
            getScript: function(t, n) {
                e.ajax({
                    url: t,
                    dataType: "script",
                    cache: !0,
                    success: n
                })
            },
            updateShareTool: function(e) {
                var t = e || this.playlist.videos[0], n = this.$videoPlayer.find(".sharetools"), r = '<iframe width="480" height="373" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" id="nyt_video_player" title="New York Times Video - Embed Player" src="http://graphics8.nytimes.com/bcvideo/1.0/iframe/embed.html?videoId=' + t.id + '&playerType=embed"></iframe>', i = function() {
                    n.data("title", t.headline), n.data("description", t.summary), n.data("url", t.publish_url || "http://www.nytimes.com/video")
                };
                n.removeClass("sharetools-init").empty(), i(), this.shareTools = new o({
                    el: n,
                    showAllModalSettings: {
                        tailDirection: "down-right",
                        positionTailSide: !1,
                        autoPosition: !1,
                        toggleSpeed: 1,
                        addlClasses: "theme-compact",
                        template: "showAllSharetoolCompactModal",
                        ad: !0
                    },
                    showAllModalData: {
                        shares: "email|Email,google|Google+,reddit|Reddit,facebook|Facebook,linkedin|LinkedIn,pinterest|Pinterest,twitter|Twitter,tumblr|Tumblr,ad",
                        embedCode: r
                    }
                })
            },
            formatDuration: function(e) {
                var t = e / 1e3, n = Math.floor(t / 60), r = Math.round(t%60);
                return r < 10 && (r = "0" + r), n + ":" + r
            },
            setMoreVideosLink: function() {
                var e = this.element.find(".section-header-link"), t = e.find("span");
                e.attr("href", this.options.moreVideos), t.html(this.options.name)
            }
        }
    }(u), u
});
