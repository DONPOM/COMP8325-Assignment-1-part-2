var menuHoverIn = function()
{
    $('#menuExtra').addClass('active');
    $('#menuDD').show();
}
var menuHoverOut = function()
{
    $('#menuExtra').removeClass('active');
    setTimeout(function() {
        if (!$('#menuExtra').hasClass('active'))
            $('#menuDD').hide();
    }, 50);
}
var newsSeeMore = function(e)
{
    e.preventDefault();
    if ($('#newsLoading').is(":visible"))
        return false;
    $this = $(this);
    $('#newsSeeMoreText').hide();
    $('#newsLoading').show();
    var page = $this.attr('p');
    $.get('//www.patreon.com/newsNext?p=' + page, function(data) {
        if (!data.trim())
            $this.hide();
        $innerBox = $('#newsBox');
        $innerBox.css({
            height: Math.max($innerBox.height(), 474)
        });
        $this.attr('p', parseInt(page) + 1);
        $('#newsLoading').hide();
        $('#newsSeeMoreText').show();
        $this.before(data);
        $('#newsBox').find('.newsDate').localTimeFromUTC("mmm d, yyyy 'at' h:MM tt");
        $innerBox.jScrollPane();
    });
}
var boxWrapperHoverIn = function()
{
    var self = $(this);
    self.find('.btnInfo').hide().removeClass('hiddenClose').fadeIn(150);
    self.find('.imageOver:first').addClass('transparent');
}
var boxWrapperHoverOut = function()
{
    var self = $(this);
    self.find('.btnInfo').addClass('hiddenClose');
    self.find('.imageOver:first').removeClass('transparent');
}
var boxOptionHoverIn = function()
{
    var self = $(this);
    self.css("background-color", "#dfe7f3");
}
var boxOptionHoverOut = function()
{
    var self = $(this);
    self.css("background-color", "#fff");
}
var seeMoreClick = function(e)
{
    e.preventDefault();
    $('#seeMore').addClass('hiddenDisplay');
    $container.infinitescroll('retrieve');
}
var followClick = function(e)
{
    e.preventDefault();
    if (!runUserCheck())
        return false;
    $this = $(this);
    if ($this.hasClass('following'))
    {
        $this.removeClass('following').find('span:first').html('Follow');
        follow = '0';
    } else
    {
        $this.addClass('following').find('span:first').html('Followed');
        follow = '1';
    }
    $.post('http://www.patreon.com/processFollow?follow=' + follow + '&uid=' + $this.attr('uid'));
}
if (typeof(window.like_target) != 'function') {
    window.like_target = function(target_id) {}
}
var actionClick = function(actionType, whichPage) {
    return function(e) {
        e.preventDefault();
        if (!runUserCheck())
            return false;
        whichPage = typeof whichPage !== 'undefined' ? whichPage : 1;
        var $shareSelector, $countSelector;
        var onSet, classname;
        switch (whichPage)
        {
        case 1:
            $shareSelector = $(this).closest('.box');
            $countSelector = $shareSelector.find('.likeCount:first');
            break;
        case 2:
            $shareSelector = $(this).closest('#spLeftButtons');
            $countSelector = $('.likeCount:first');
            break;
        case 3:
            $shareSelector = $(this).closest('#spLeftButtons');
            var hidNum = $shareSelector.attr('hid');
            var $shareBox = $('#' + hidNum);
            $countSelector = $shareBox.find('.likeCount:first');
            switch (actionType)
            {
            case 1:
                classname = "uLike";
                break;
            default:
                return false;
            }
            if ($shareBox.hasClass(classname))
                $shareBox.removeClass(classname);
            else
                $shareBox.addClass(classname);
            break;
        default:
            break;
        }
        switch (actionType)
        {
        case 1:
            classname = "uLike";
            break;
        default:
            return false;
        }
        if ($shareSelector.hasClass(classname))
        {
            onSet = '0';
            $shareSelector.removeClass(classname);
            if (actionType == 1)
            {
                var finalCount = parseInt($countSelector.html()) - 1;
                if (finalCount > 0)
                {
                    $countSelector.html(finalCount);
                    $('#svBottomRight .likeCount:first').html(finalCount);
                } else
                {
                    $countSelector.html('');
                    $('#svBottomRight .likeCount:first').html('');
                }
            }
        } else
        {
            onSet = '1';
            $shareSelector.addClass(classname);
            if (actionType == 1)
            {
                if ($countSelector.text())
                {
                    var finalCount = parseInt($countSelector.text()) + 1;
                    $countSelector.html(finalCount);
                    $('#svBottomRight .likeCount:first').html(finalCount);
                } else
                {
                    $countSelector.html('1');
                    $('#svBottomRight .likeCount:first').html('1');
                }
            }
        }
        $.post('http://www.patreon.com/processAction?ty=' + actionType + '&on=' + onSet + '&hid=' + $shareSelector.attr('hid'));
        return false;
    }
}
var chatIconClick = function(e)
{
    e.preventDefault();
    var $chatSelector = $(this).closest('.boxWrapper').find('.comments:first');
    var $commentsHide = $chatSelector.find('.commentsHide:first');
    if ($(this).hasClass("active")) {
        $(this).removeClass("active");
        $chatSelector.addClass('hiddenDisplay');
        $commentsHide.addClass('hiddenDisplay');
        $container.isotope('reLayout');
    } else {
        $(this).addClass("active");
        $chatSelector.removeClass('hiddenDisplay');
        $commentsHide.removeClass('hiddenDisplay');
        $chatSelector.find('#inputComment').focus();
        $container.isotope('reLayout');
    }
    return false;
}
var commentFormSubmit = function(e) {
    e.preventDefault();
    if (!runUserCheck())
        return false;
    var commentFormCB = function($form) {
        return function(data) {
            $form.parent().prev().children('.closeX').attr('cid', data).click(deleteCommentClick).hover(commentHoverHoverIn, commentHoverHoverOut);
            $form.parent().prev('.commentHover.isOwner').hover(commentHoverHoverIn, commentHoverHoverOut);
        }
    }
    if ($(this).find('#inputComment').val() == '')
        return false;
    $(this).children('#commentSubmitBtn').addClass("disabled").attr('disabled', 'disabled');
    var $form = $(this), postHid = $form.children('#hid').val(), postThread = $form.children('#threadid').val(), postComment = $form.find('#inputComment').val(), threadClass = '', threadStyle = '';
    if (postThread)
    {
        var prevMargin = parseInt($form.closest('.commentBox').prev().css('margin-left'));
        threadClass = ' thread';
        threadStyle = ' style="margin-left:' + (prevMargin + 10) + 'px;"';
    }
    botFieldData = $('#commentField2').val();
    $.post('/processComment', {
        hid: postHid,
        threadid: postThread,
        inputComment: postComment,
        notABot: true,
        botCheck: botFieldData
    }, commentFormCB($form));
    $form.parent().before('<div class="commentBox commentHover isOwner' + threadClass + '"' + threadStyle + '><a class="closeX posAbsolute hiddenClose" href="#"></a><div class="boxUserImage small"><a href="/user?u=' + myUID + '" ><img src="' + myThumbUrl + '" /></a></div><div class="commentContent"><p class="commenter urlBreak"><a href="#">' + myName + '</a> ' + postComment + '</p><p class="commentDate"><span class="dateRun">Just now</span></p></div><div class="clearfix"></div></div>');
    if (postThread)
    {
        $form.closest('.commentBox').remove();
    } else
    {
        $form.find('#inputComment').val('').change();
        $form.find('#commentSubmitBtn').removeClass("disabled").removeAttr('disabled');
    }
    $container.isotope('reLayout');
    return false;
}
var commentHoverHoverIn = function() {
    $(this).children(".closeX").removeClass('hiddenClose');
}
var commentHoverHoverOut = function() {
    $(this).children(".closeX").addClass('hiddenClose');
}
var deleteCommentClick = function(e) {
    e.preventDefault();
    var clickedComment = $(this).attr('cid');
    if (clickedComment == undefined)
        return false;
    var $commentBox = $(this).parent('.commentBox');
    apprise('Are you sure you want to delete this comment? You cannot undo once deleted.', {
        'verify': true
    }, function(r)
    {
        if (r)
        {
            $.post('/processCommentDelete?cid=' + clickedComment);
            $commentBox.remove();
            $container.isotope('reLayout');
        } else
        {}
    });
    return false;
}
var replyCommentClick = function(e) {
    e.preventDefault();
    var clickedComment = $(this).attr('cid');
    if (clickedComment == undefined)
        return false;
    var clickedHid = $(this).closest('.box').attr('hid');
    if (clickedHid == undefined)
        clickedHid = $(this).closest('#popWrapper').attr('hid');
    if (clickedHid == undefined)
        clickedHid = $(this).closest('#sharePage').attr('hid');
    if (clickedHid == undefined)
        return false;
    var $commentBox = $(this).closest('.commentBox');
    if ($commentBox.next('.commentBox').find('#threadid').val() != clickedComment)
    {
        var insertBox = '<div class="commentBox"><div class="boxUserImage small"><a href="#"><img src="' + myThumbUrl + '"></a></div><form id="commentForm" action="#" method="post"><input type="hidden" id="threadid" name="threadid" value="' + clickedComment + '"><input type="hidden" id="hid" name="hid" value="' + clickedHid + '"><div class="commentContentInput"><textarea id="inputComment" name="inputComment" class="commentInput"></textarea></div><input id="commentSubmitBtn" name="commentSubmitBtn" value="Post" type="submit" class="btn submitBtn bottomRight"></form><div class="clearfix"></div></div>';
        $commentBox.after(insertBox).next().find('.commentInput:first').watermark('Write a comment...', {
            useNative: false
        }).elastic().focus().end().find("#commentForm").submit(commentFormSubmit);
    }
    $container.isotope('reLayout');
    return false;
}
var viewCommentsClick = function(e) {
    e.preventDefault();
    $(this).parent().parent().parent().hide().siblings().removeClass('hiddenDisplay');
    $container.isotope('reLayout');
    return false;
}
var inviteOneSubmit = function(e) {
    e.preventDefault();
    var $buttons = $(this).find('.btn:first');
    $buttons.addClass("disabled").attr('disabled', 'disabled');
    var sid = $('#inviteSid').val(), uid = $(this).attr('uid');
    if (sid != 0)
    {
        $.post('http://www.patreon.com/processInviteOne', {
            s: sid,
            uid: uid
        });
        $.fancybox.close();
    } else
    {
        apprise('Please select a photoshoot first.');
    }
    $buttons.removeClass("disabled").removeAttr('disabled');
    return false;
}
var shareEditSubmit = function(e) {
    e.preventDefault();
    var editTitle = $('#seShareTitle').val();
    var editText = $('#seText').val();
    var editPatrons = parseInt($('#sePrivate').val());
    var $seButtons = $(this).children('#seInner').find('.btn');
    $seButtons.addClass("disabled").attr('disabled', 'disabled');
    var $form = $(this), editHid = $(this).closest('#shareEdit').attr('hid');
    $.post('http://www.patreon.com/processShareEdit', {
        hid: editHid,
        seShareTitle: editTitle,
        seText: editText,
        sePrivate: editPatrons
    });
    var setTitle = '';
    if (editTitle)
        setTitle = '<h1>' + editTitle + '</h1>';
    if (editText ||!$('#' + editHid).hasClass('photo'))
    {
        $('#' + editHid).find('.shareText:first').html(setTitle + '<div class="shareContent">' + editText + '</div>').show();
        $('#' + editHid).find('.imagePopup:first').attr('original-title', '');
        $('#' + editHid).find('.utitle:first').text(editTitle);
    } else
    {
        $('#' + editHid).find('.shareText:first').html('').hide();
        $('#' + editHid).find('.imagePopup:first').attr('original-title', editTitle);
        $('#' + editHid).find('.utitle:first').text(editTitle);
    }
    if (editPatrons > 0)
    {
        $('#' + editHid).attr('prv', editPatrons).find('.tabTri:first').removeClass('hiddenDisplay').attr('original-title', $('#sePrivate option:selected').text());
    } else
    {
        $('#' + editHid).attr('prv', '0').find('.tabTri:first').addClass('hiddenDisplay');
    }
    if ($container)
    {
        $container.isotope('reLayout');
    }
    $.fancybox.close();
    return false;
}
var shareDeleteClick = function(e) {
    e.preventDefault();
    var deleteHid = $(this).closest('#shareEdit').attr('hid');
    if (deleteHid == undefined)
        return false;
    apprise('Are you sure you want to delete this item? You cannot undo once deleted.', {
        'verify': true
    }, function(r)
    {
        if (r)
        {
            $.post('http://www.patreon.com/processShareDelete?hid=' + deleteHid, function(data) {
                if (($container&&!$container.length) ||!$container)
                {
                    window.top.location.href = 'http://www.patreon.com/user?ty=a&delete=1';
                    return false;
                }
            });
            $.fancybox.close();
            $('#' + deleteHid).remove();
            $('#hid' + deleteHid).remove();
            if ($container)
            {
                $container.isotope('reLayout');
            }
        } else
        {}
    });
    return false;
}
var listElemHoverIn = function() {
    var self = $(this);
    self.addClass('selected');
}
var listElemHoverOut = function() {
    var self = $(this);
    self.removeClass('selected');
}
var checkLike = function(data) {
    if ($('#likesLoading').length) {
        $likesLoading = $('#likesLoading');
        $likesLoading.hide();
        $likesLoading.after(data);
        $likesLoading.nextAll(".userThumb").tipsy({
            fade: true,
            gravity: 's'
        });
    } else {
        setTimeout(function() {
            checkLike(data);
        }, 1000);
    }
}
var checkPledge = function(data) {
    if ($('#pledgesLoading').length) {
        $pledgesLoading = $('#pledgesLoading');
        $pledgesLoading.hide();
        $pledgesLoading.after(data);
        $pledgesLoading.nextAll(".userThumb").tipsy({
            fade: true,
            gravity: 's'
        });
    } else {
        setTimeout(function() {
            checkPledge(data);
        }, 1000);
    }
}
function registerAllItemBinds($boxWrappers)
{
    $boxWrappers.find('.dateBox').localTimeFromUTC("mmm d, yyyy 'at' h:MM tt");
    $boxWrappers.find('.dateRun').localTimeFromUTC("mmm d, yyyy 'at' h:MM tt");
    $boxWrappers.hover(boxWrapperHoverIn, boxWrapperHoverOut);
    $boxWrappers.find('a.imagePopup').tipsy({
        fade: true,
        gravity: 'n',
        offset: 16
    });
    $boxWrappers.find('.showPopup').tipsy({
        fade: true,
        gravity: 's'
    });
    $boxWrappers.find('a.btnLearn').click(actionClick(1));
    $boxWrappers.find('a.btnChat').tipsy({
        fade: true,
        gravity: 's'
    }).click(chatIconClick);
    $boxWrappers.find('a.btnEdit').tipsy({
        fade: true,
        gravity: 's'
    }).fancybox({
        'type': 'inline',
        'autoScale': false,
        'scrolling': 'no',
        'showNavArrows': false,
        'titleShow': false,
        'onStart': function(items, index) {
            var clickedHid = $(items[index]).closest('.box').attr('hid');
            editStart(clickedHid, items, index);
        },
        'onClosed': function() {
            editClose();
        }
    });
    var $commentInput = $boxWrappers.find('.commentInput');
    $commentInput.watermark('Write a comment...', {
        useNative: false
    }).elastic();
    $boxWrappers.find("form#commentForm").submit(commentFormSubmit);
    $boxWrappers.find(".commentHover.isOwner").hover(commentHoverHoverIn, commentHoverHoverOut);
    $boxWrappers.find(".commentBox .closeX").click(deleteCommentClick);
    $boxWrappers.find('.viewCommentsLink').click(viewCommentsClick);
    $boxWrappers.find(".reply").click(replyCommentClick);
    var moreText = "See more";
    var lessText = "See Less";
    var adjustheight;
    $boxWrappers.find(".shareText").each(function() {
        adjustheight = $(this).css('min-height');
        if (!adjustheight || adjustheight == '0px')
            adjustheight = 296;
        else
            adjustheight = parseInt(adjustheight, 10) + 46;
        if ($(this).prop('scrollHeight') > adjustheight)
            $(this).parent().append('<a class="moreText" href="#">See more</a>');
    });
    $boxWrappers.find(".moreText").toggle(function() {
        $(this).parents("div:first").find(".shareText:first").css('max-height', 'none').css('overflow', 'visible');
        $(this).text(lessText);
        $container.isotope('reLayout');
    }, function() {
        var adjustheight = $(this).parents("div:first").find(".shareText:first").css('min-height');
        if (!adjustheight || adjustheight == '0px')
            adjustheight = 250;
        else
            adjustheight = parseInt(adjustheight, 10);
        $(this).parents("div:first").find(".shareText:first").css('max-height', adjustheight).css('overflow', 'hidden');
        $(this).text(moreText);
        $container.isotope('reLayout');
    });
    $boxWrappers.prev('.showPopup').tipsy({
        fade: true,
        gravity: 's'
    });
}
function registerAllItemBindsMin($boxWrappers)
{
    $boxWrappers.hover(boxWrapperHoverIn, boxWrapperHoverOut);
    $boxWrappers.find('.favesHover').hover(function() {
        $(this).parent().find('.captionCenter:first').stop(true, true).fadeOut(200);
    }, function() {
        $(this).parent().find('.captionCenter:first').stop(true, true).fadeIn(50);
    });
    $boxWrappers.find('.btnFollow').click(followClick);
}
function editStart(clickedHid, items, index)
{
    var $shareBox = $('#' + clickedHid);
    var shareImage = $shareBox.find('.imagePopup:first img:first').attr('src');
    var $shareEdit = $('#shareEdit');
    var private = parseInt($shareBox.attr('prv'));
    $('#fancybox-left').hide();
    $('#fancybox-right').hide();
    $('#seShareTitle').val(stripTags($shareBox.find('.utitle:first').text()));
    $('#seText').val(stripTags($shareBox.find('.shareContent:first').text()));
    if (shareImage)
        $('#seImage').css('background', 'url(' + shareImage + ') no-repeat 50%').parent().css('visibility', 'visible');
    else
        $('#seImage').parent().css('visibility', 'hidden');
    $('#sePrivate').val(private);
    $shareEdit.attr('hid', clickedHid);
}
function editClose()
{
    var $shareEdit = $('#shareEdit');
    $shareEdit.removeAttr('hid');
    if ($container)
    {
        $container.infinitescroll('resume');
    }
}(function($) {
    $.fn.localTimeFromUTC = function(format) {
        return this.each(function() {
            var currentDate = new Date();
            var tagText = $(this).html();
            var givenDate = new Date(tagText + ' UTC');
            var diff = (currentDate.getTime() - givenDate.getTime()) / 1000, day_diff = Math.floor(diff / 86400);
            if (isNaN(day_diff))
                return;
            if (day_diff > 0)
            {
                $(this).html(givenDate.format(format));
            } else
            {
                $(this).html(getTimeAgo(diff));
            }
        });
    };
})(jQuery);
function getTimeAgo(diff) {
    return (diff < 60 && "Just now" || diff < 120 && "1 minute ago" || diff < 3600 && Math.floor(diff / 60) + " minutes ago" || diff < 7200 && "1 hour ago" || diff < 86400 && Math.floor(diff / 3600) + " hours ago");
}
function stripTags(html)
{
    var tmp = document.createElement("DIV");
    tmp.innerHTML = html;
    return tmp.textContent || tmp.innerText;
}
function runUserCheck()
{
    if (!myUID)
    {
        $('#firstmenuitem a').click();
        return false;
    }
    return true;
}
!function($) {
    "use strict"
    var transitionEnd
    $(document).ready(function() {
        $.support.transition = (function() {
            var thisBody = document.body || document.documentElement, thisStyle = thisBody.style, support = thisStyle.transition !== undefined || thisStyle.WebkitTransition !== undefined || thisStyle.MozTransition !== undefined || thisStyle.MsTransition !== undefined || thisStyle.OTransition !== undefined
            return support
        })()
        if ($.support.transition) {
            transitionEnd = "TransitionEnd"
            if ($.browser.webkit) {
                transitionEnd = "webkitTransitionEnd"
            } else if ($.browser.mozilla) {
                transitionEnd = "transitionend"
            } else if ($.browser.opera) {
                transitionEnd = "oTransitionEnd"
            }
        }
    })
    var Alert = function(content, options) {
        this.settings = $.extend({}, $.fn.alert.defaults, options)
        this.$element = $(content).delegate(this.settings.selector, 'click', this.close)
    }
    Alert.prototype = {
        close: function(e) {
            var $element = $(this).parent('.alert-message')
            e && e.preventDefault()
            $element.removeClass('in')
            function removeElement() {
                $element.hide()
            }
            $.support.transition && $element.hasClass('fade') ? $element.bind(transitionEnd, removeElement) : removeElement()
        }
    }
    $.fn.alert = function(options) {
        if (options === true) {
            return this.data('alert')
        }
        return this.each(function() {
            var $this = $(this)
            if (typeof options == 'string') {
                return $this.data('alert')[options]()
            }
            $(this).data('alert', new Alert(this, options))
        })
    }
    $.fn.alert.defaults = {
        selector: '.close'
    }
    $(document).ready(function() {
        new Alert($('body'), {
            selector: '.alert-message[data-alert] .close'
        })
    })
}(window.jQuery || window.ender);
var shareImgList;
var shareImgListIndex = 0;
var hasClosedShareUrl = false;
var isShareSet = false;
var $container;
$(document).ready(function() {
    $('#menuContainer .hoverShow').hover(menuHoverIn, menuHoverOut);
    $("a.loginPop:first").fancybox({
        'overlayColor': '#FFF',
        'type': 'iframe',
        'autoScale': false,
        'autoDimensions': false,
        'width': 420,
        'height': 325,
        'scrolling': 'no',
        'showNavArrows': false
    });
    switch (osPage) {
    case'index':
        runIndex();
        break;
    case'home':
        runHome();
        break;
    case'user':
        runUser();
        break;
    case'share':
        runShare();
        break;
    case'upload':
        runPatron();
        break;
    default:
        break;
    }
    $("#userMsgAll").fancybox({
        'overlayColor': '#FFF',
        'type': 'iframe',
        'autoScale': false,
        'autoDimensions': false,
        'modal': false,
        'width': 800,
        'height': 600,
        'scrolling': 'no',
        'showNavArrows': false,
        'onClosed': function() {
            $("body").css('overflow', 'visible');
            if (typeof $container !== 'undefined')
                $container.infinitescroll('resume');
        },
        'onStart': function(items, index) {
            if (typeof $container !== 'undefined')
                $container.infinitescroll('pause');
            $("body").css('overflow', 'hidden');
        }
    }).click(function() {
        $(this).find('.hasNew:first').addClass('hiddenDisplay')
    });
});
function runHome()
{
    $container = $('#boxGrid');
    if ($.browser.mozilla)
    {
        $container.isotope({
            itemSelector: '.box',
            transformsEnabled: false,
            getSortData: {
                topOrder: function($item) {
                    if ($item.attr('si'))
                        return parseInt($item.attr('si'));
                    return 0;
                }
            },
            sortBy: 'topOrder',
            masonry: {
                columnWidth: 362
            }
        });
    } else
    {
        $container.isotope({
            itemSelector: '.box',
            getSortData: {
                topOrder: function($item) {
                    if ($item.attr('si'))
                        return parseInt($item.attr('si'));
                    return 0;
                }
            },
            sortBy: 'topOrder',
            masonry: {
                columnWidth: 362
            }
        });
    }
    if (noScroll)
    {
        var $boxWrappers = $('.boxWrapper');
        registerAllItemBinds($boxWrappers);
    } else
    {
        $container.infinitescroll({
            navSelector: '#page_nav',
            nextSelector: '#page_nav a',
            itemSelector: '.box',
            loading: {
                finishedMsg: 'No more items to load.',
                msgText: "Loading the next set of posts...",
                img: '/images/inf_load.gif'
            },
            errorCallback: function() {}
        }, function(newElements) {
            var $newElems = $(newElements);
            var $boxWrappers = $newElems.children('.boxWrapper');
            if (!noBind)
                registerAllItemBinds($boxWrappers);
            $container.isotope('appended', $newElems, function() {});
        });
        var $boxWrappers = $('.boxWrapper');
        if (!noBind)
            registerAllItemBinds($boxWrappers);
    }
    $("#startPopup").fancybox({
        'overlayColor': '#FFF',
        'type': 'iframe',
        'autoScale': false,
        'autoDimensions': false,
        'modal': false,
        'width': 800,
        'height': 280,
        'scrolling': 'no',
        'showNavArrows': false,
        'onClosed': function() {
            $("body").css('overflow', 'visible');
        },
        'onStart': function(items, index) {
            $("body").css('overflow', 'hidden');
        }
    });
    $("#profileLeft .showPopup").tipsy({
        fade: true,
        gravity: 's'
    });
    $("#mainCategories .showPopup").tipsy({
        fade: true,
        gravity: 's'
    });
    $().UItoTop({
        easingType: 'easeOutQuart',
        scrollSpeed: 400
    });
}
function runIndex()
{}
function runUser()
{
    if (!noScroll)
    {
        $container = $('#boxGrid');
        if ($.browser.mozilla)
        {
            $container.isotope({
                itemSelector: '.box',
                transformsEnabled: false,
                getSortData: {
                    topOrder: function($item) {
                        if ($item.attr('si'))
                            return parseInt($item.attr('si'));
                        return 0;
                    }
                },
                sortBy: 'topOrder',
                masonry: {
                    columnWidth: 362
                }
            });
        } else
        {
            $container.isotope({
                itemSelector: '.box',
                getSortData: {
                    topOrder: function($item) {
                        if ($item.attr('si'))
                            return parseInt($item.attr('si'));
                        return 0;
                    }
                },
                sortBy: 'topOrder',
                masonry: {
                    columnWidth: 362
                }
            });
        }
    }
    if (noScroll)
    {
        var $boxWrappers = $('.box.option .boxWrapper');
        $boxWrappers.hover(boxOptionHoverIn, boxOptionHoverOut);
        $('#followBtn').click(followClick);
    } else
    {
        $container.infinitescroll({
            navSelector: '#page_nav',
            nextSelector: '#page_nav a',
            itemSelector: '.box',
            loading: {
                finishedMsg: 'No more items to load.',
                msgText: "Loading the next set of posts...",
                img: '/images/inf_load.gif'
            },
            errorCallback: function() {}
        }, function(newElements) {
            var $newElems = $(newElements);
            var $boxWrappers = $newElems.children('.boxWrapper');
            registerAllItemBinds($boxWrappers);
            $container.isotope('appended', $newElems, function() {});
        });
        var $boxWrappers = $('.boxWrapper');
        registerAllItemBinds($boxWrappers);
    }
    $("#seeFollowers,#seeFollowing").fancybox({
        'overlayColor': '#FFF',
        'type': 'iframe',
        'autoScale': false,
        'autoDimensions': false,
        'width': 656,
        'height': 600,
        'scrolling': 'no',
        'showNavArrows': false,
        'onClosed': function() {
            if (typeof $container !== 'undefined')
                $container.infinitescroll('resume');
        },
        'onStart': function(items, index) {
            if (typeof $container !== 'undefined')
                $container.infinitescroll('pause');
        }
    });
    $("#profileLeft .showPopup").tipsy({
        fade: true,
        gravity: 's'
    });
    $("#mainCategories .showPopup").tipsy({
        fade: true,
        gravity: 's'
    });
    $("#boxGrid .showPopup").tipsy({
        fade: true,
        gravity: 's'
    });
    $("#rightButtons a").tipsy({
        fade: true,
        gravity: 's'
    });
    $().UItoTop({
        easingType: 'easeOutQuart',
        scrollSpeed: 400
    });
    $('#seDelete').click(shareDeleteClick);
    $("#seForm").submit(shareEditSubmit);
    $("#inviteForm").submit(inviteOneSubmit);
    if (myUID)
    {
        $("#newsSeeMore,#messageBtn").fancybox({
            'overlayColor': '#FFF',
            'type': 'iframe',
            'autoScale': false,
            'autoDimensions': false,
            'modal': false,
            'width': 800,
            'height': 600,
            'scrolling': 'no',
            'showNavArrows': false,
            'onClosed': function() {
                $("body").css('overflow', 'visible');
                if (typeof $container !== 'undefined')
                    $container.infinitescroll('resume');
            },
            'onStart': function(items, index) {
                if (typeof $container !== 'undefined')
                    $container.infinitescroll('pause');
                $("body").css('overflow', 'hidden');
            }
        });
    } else
        $("#newsSeeMore,#messageBtn").click(function() {
            $('#firstmenuitem a').click();
            return false;
        });
}
function runPatron()
{
    $container = $('#boxGrid');
    $container.isotope({
        itemSelector: '.box',
        getSortData: {
            topOrder: function($item) {
                if ($item.attr('si'))
                    return parseInt($item.attr('si'));
                return 0;
            }
        },
        sortBy: 'topOrder',
        masonry: {
            columnWidth: 362
        }
    });
    if (!noScroll)
    {
        $container.infinitescroll({
            navSelector: '#page_nav',
            nextSelector: '#page_nav a',
            itemSelector: '.box',
            loading: {
                finishedMsg: 'No more items to load.',
                msgText: "Loading the next set of posts...",
                img: '/images/inf_load.gif'
            },
            errorCallback: function() {}
        }, function(newElements) {
            var $newElems = $(newElements);
            var $boxWrappers = $newElems.children('.boxWrapper');
            registerAllItemBinds($boxWrappers);
            $container.isotope('appended', $newElems, function() {});
        });
        var $boxWrappers = $('.boxWrapper');
        registerAllItemBinds($boxWrappers);
    } else
    {
        $('.boxWrapper').hover(boxWrapperHoverIn, boxWrapperHoverOut).find('.showPopup').tipsy({
            fade: true,
            gravity: 's'
        });
    }
    $("#profileLeft .showPopup").tipsy({
        fade: true,
        gravity: 's'
    });
    $("#boxGrid .showPopup").tipsy({
        fade: true,
        gravity: 's'
    });
    $().UItoTop({
        easingType: 'easeOutQuart',
        scrollSpeed: 400
    });
    $('#seDelete').click(shareDeleteClick);
    $("#seForm").submit(shareEditSubmit);
    if (myUID)
    {
        $("#newsSeeMore,#messageBtn").fancybox({
            'overlayColor': '#FFF',
            'type': 'iframe',
            'autoScale': false,
            'autoDimensions': false,
            'modal': false,
            'width': 800,
            'height': 600,
            'scrolling': 'no',
            'showNavArrows': false,
            'onClosed': function() {
                $("body").css('overflow', 'visible');
                $container.infinitescroll('resume');
            },
            'onStart': function(items, index) {
                $container.infinitescroll('pause');
                $("body").css('overflow', 'hidden');
            }
        });
    } else
        $("#newsSeeMore,#messageBtn").click(function() {
            $('#firstmenuitem a').click();
            return false;
        });
}
function runShare()
{
    $container = $();
    $('#shareDate').localTimeFromUTC("mmm d, yyyy 'at' h:MM tt");
    console.log('attaching action to button ' + 1);
    $('a.btnLearn:first').click(actionClick(1, 2));
    var $commentInput = $('#inputComment');
    var $commentsDiv = $('#spComments');
    $commentsDiv.find(".dateRun").localTimeFromUTC("mmm d, yyyy 'at' h:MM tt");
    $("#profileLeft .showPopup").tipsy({
        fade: true,
        gravity: 's'
    });
    $("#spBottomLeft .showPopup").tipsy({
        fade: true,
        gravity: 's'
    });
    $(".userThumb").tipsy({
        fade: true,
        gravity: 's'
    });
    $("#spLeftButtons .popup").tipsy({
        fade: true,
        gravity: 's'
    });
    $('#seDelete').click(shareDeleteClick);
    $(".editAdmin:first").fancybox({
        'type': 'inline',
        'autoScale': false,
        'scrolling': 'no',
        'showNavArrows': false,
        'titleShow': false,
        'onStart': function(items, index) {
            var clickedHid = $(items[index]).attr('hid');
            editStart(clickedHid, items, index);
        },
        'onClosed': function() {
            editClose();
        }
    });
}
function noResubmit($form)
{
    $form.submit(function() {
        return false;
    });
    return true;
}
