
var QQ = {};
QQ.Cookie={
    set:function(name,value,expires,path,domain){
        if(typeof expires=="undefined"){
            expires=new Date(new Date().getTime()+3600*1000);
        }
        document.cookie=name+"="+escape(value)+((expires)?"; expires="+expires.toGMTString():"")+((path)?"; path="+path:"; path=/")+((domain)?";domain="+domain:"");
    },
    get:function(name){
        var arr=document.cookie.match(new RegExp("(^| )"+name+"=([^;]*)(;|$)"));
        if(arr!=null){
            return unescape(arr[2]);
        }
        return null;
    },
    clear:function(name,path,domain){
        if(this.get(name)){
            document.cookie=name+"="+((path)?"; path="+path:"; path=/")+((domain)?"; domain="+domain:"")+";expires=Fri, 02-Jan-1970 00:00:00 GMT";
        }
    }
};
QQ.localData = {
    userData: null,
    name: location.hostname,
    isLocalStorage: typeof localStorage == 'undefined'?false:true,
    init: function(){
        if (!this.userData) {
            try {
                this.userData = document.createElement('INPUT');
                this.userData.type = "hidden";
                this.userData.style.display = "none";
                this.userData.addBehavior ("#default#userData");
                document.body.appendChild(this.userData);
                var expires = new Date();
                expires.setDate(expires.getDate()+365);
                this.userData.expires = expires.toUTCString();
            } catch(e) {
                return false;
            }
        }
        return true;
    },
    set: function(key, value) {
        if(this.isLocalStorage){
            localStorage.setItem(key, value);
        }
        else if(this.init()){
            this.userData.load(this.name);
            this.userData.setAttribute(key, value);
            this.userData.save(this.name);
        }
    },
    get: function(key) {
        if(this.isLocalStorage){
            return localStorage.getItem(key);
        }
        else if(this.init()){
            this.userData.load(this.name);
            return this.userData.getAttribute(key)
        }
        return null;
    },
    remove: function(key) {
        if(this.isLocalStorage){
            localStorage.removeItem(key);
        }
        else if(this.init()){
            this.userData.load(this.name);
            this.userData.removeAttribute(key);
            this.userData.save(this.name);
        }
    }
};
QQ.LoadScript = function(url, callback, charset){
    var script = document.createElement("script");
    script.type = "text/javascript";
    if(charset){
        script.setAttribute("charset",charset);
    }
    if (script.readyState){
        script.onreadystatechange = function(){
            if(script.readyState == "loaded" || script.readyState == "complete"){
                script.onreadystatechange = null;
                callback();
                document.getElementsByTagName("head")[0].removeChild(this);
            }
        };
    }
    else {
        script.onload = function(){
            callback();
            document.getElementsByTagName("head")[0].removeChild(this);
        };
    }
    script.src = url;
    document.getElementsByTagName("head")[0].appendChild(script);
}

/*Ajax*/
var DomainProxy = function(){
    this.construct.apply(this, arguments);
};
DomainProxy.contentWin = {};
DomainProxy.prototype = {
    construct: function(options){
        this.domain = options.domain;
        this.url = options.url;
        this.callfunc = options.callfunc || function(){};
        this.setDomain();
        if(DomainProxy.contentWin[this.url]){
            this.callfunc(DomainProxy.contentWin[this.url]);
        }else{
            this.create();
        }
    },
    setDomain: function(){
        try{
            if(document.domain != this.domain)
                document.domain = this.domain;
        }catch(e){}
    },
    addEvent: function(node, type, listener, useCapture){
        if(node.addEventListener){
            node.addEventListener(type, listener, useCapture);
            return true;
        }else if(node.attachEvent){
            return node.attachEvent('on'+type, listener);
        }
        else return false;
    },
    create: function(){
        var that = this;
        this.iframe = document.createElement('iframe');
        this.iframe.style.display = 'none';
        document.body.appendChild( this.iframe );
        this.addEvent(this.iframe, 'load', function(){
            DomainProxy.contentWin[that.url] = that.iframe.contentWindow;
            that.callfunc( that.iframe.contentWindow );
        }, false);
        this.iframe.src = this.url;
    }
};
var Ajax = function(){
    this.construct.apply(this, arguments);
};
Ajax.prototype = {
    construct: function(options){
        this.xmlhttp = this.getXmlHttpObject(options.domainProxy || window);
        this.options = options || {};
        this.headers = this.options.headers || {};
        this.method = (this.options.method || "get").toLowerCase();
        this.async = this.options.async || true;
        this.url = this.options.url || null;
        this.postdata = this.options.postdata || "";
        this.timeout = null;

        if (this.method == "post")
            this.headers["Content-Type"] = "application/x-www-form-urlencoded";
    },
    start: function(){
        var instance = this;
        var requestParams = typeof(this.options.parameters) == "string" ? this.options.parameters : this.encodeParameters(this.options.parameters || {});
        if (requestParams && this.method == "get")
            this.url += (this.url.indexOf('?') > -1 ? '&' : '?') + requestParams;
        this.xmlhttp.open(this.method, this.url, this.async);
        this.onLoading();

        for (header in this.headers) {
            if (!this.headers.hasOwnProperty(header))
                continue;
            this.xmlhttp.setRequestHeader(header, this.headers[header]);
        }

        this.xmlhttp.onreadystatechange = function(){
            instance.dispatchEvent();
        }

        if (this.options.timeout && typeof(this.options.timeout) == "number")
            this.timeout = setTimeout(function(){
                instance.onAbort();
                instance.onLoaded();
                instance.onTimeout();
            }, this.options.timeout * 1000);

        this.xmlhttp.send(this.method == "post" ? (this.postdata || requestParams) : null);
        if (!this.async)
            this.dispatchEvent();
    },
    getXmlHttpObject: function(win){
        if (win.XMLHttpRequest)
            return new win.XMLHttpRequest();
        else
        if (win.ActiveXObject) {
            var msxmls = new Array("Msxml2.XMLHTTP", "Microsoft.XMLHTTP");
            for (var i = 0, len = msxmls.length; i < len; i++) {
                try {
                    return new win.ActiveXObject(msxmls[i]);
                }
                catch (e) {
                }
            }
        }
        throw new Error("Could not instantiate XmlHttpObject!");
    },
    encodeParameters: function(parameters){
        var paras = [];
        for (parameter in parameters)
            //paras.push(escape(parameter) + "=" + escape(parameters[parameter]));
            paras.push(parameter + "=" + parameters[parameter]);
        return paras.join('&');
    },
    addRequestHeader: function(name, content){
        this.headers[name] = content;
    },
    onTimeout: function(){},
    onLoading: function(){},
    onLoaded: function(){},
    onComplete: function(responseObject){},
    onError: function(errorObject){},
    onAbort: function(){
        if (this.xmlhttp)
            this.xmlhttp.abort();
    },
    dispatchEvent: function(){
        if (this.xmlhttp.readyState == 4) {
            if (this.timeout)
                clearTimeout(this.timeout);
            this.onLoaded();

            if (this.xmlhttp.status == 200)
                this.onComplete({
                    textString: this.xmlhttp.responseText,
                    xmlDocument: this.xmlhttp.responseXML
                });
            else
            if (this.xmlhttp.status == 0)
                this.onAbort();
            else
                this.onError({
                    url: this.url,
                    status: this.xmlhttp.status,
                    statusText: this.xmlhttp.statusText,
                    responseText: this.xmlhttp.responseText
                });
        }
    }
};

//频道导航更多
function navMore(){
    $("#navMore").bind("mouseover",function(){
        $("#navMore ul").show();
        $(this).addClass("navMoreHover");
    });
    $("#navMore").bind("mouseout",function(){
        $("#navMore ul").hide();
        $(this).removeClass("navMoreHover");
    });
}

//大家爱看
function djakInit(){
    $("#dajiaikanrefresh").hide();
    $("#dajiaaikanloading").show();
    QQ.LoadScript("http://www.qq.com/json_data_djak/index_data_djak/cur_data.json?random="+Math.random(), function(){
        var djakDataTempArr = []; //大家爱看临时数据
        djakDataTempArr.push('<div class="bd"><div class="imgArea"><a href="http://icare.qq.com/#'+(jsonData[0].channel_id)+'_list_'+(jsonData[0].id)+'" target="_blank"><img alt="'+jsonData[0].pub_title+'" src="'+jsonData[0].img_url+'"></a></div>');
        djakDataTempArr.push('<div class="txtArea"><dl>');
        djakDataTempArr.push('<dt><a href="http://icare.qq.com/#'+(jsonData[0].channel_id)+'_list_'+(jsonData[0].id)+'" target="_blank">'+jsonData[0].pub_title+'</a></dt>');
        djakDataTempArr.push('<dd><a class="times" href="http://icare.qq.com/#'+(jsonData[0].channel_id)+'_list_'+(jsonData[0].id)+'" target="_blank">'+jsonData[0].re_count+'次</a><a href="http://icare.qq.com/#'+(jsonData[0].channel_id)+'_list_'+(jsonData[0].id)+'_broadt" target="_blank">转播</a> | <a href="http://icare.qq.com/#'+(jsonData[0].channel_id)+'_list_'+(jsonData[0].id)+'_comt" target="_blank">评论</a></dd>');
        djakDataTempArr.push('</dl></div></div>');
        djakDataTempArr.push('<div class="ft"><ul>');
        for(var i=1;i<8;i++){
            var djakLinkUrl = 'http://icare.qq.com/#'+(jsonData[i].channel_id)+'_list_'+(jsonData[i].id);
            djakDataTempArr.push('<li><a target="_blank" href="'+djakLinkUrl+'">'+(jsonData[i].pub_title)+'</a></li>');
        }
        djakDataTempArr.push('</ul></div>');
        document.getElementById("djakContent").innerHTML = djakDataTempArr.join("");
        $("#dajiaaikanloading").hide();
        $("#dajiaikanrefresh").show();
    });
    $("#dajiaikanrefresh").click(function(){
        djakInit();
    });
}

/*星座运势*/
var astroToday = "白羊座";
var astroArr = 0;
var astroYunshi = "80";
var astroUrl = "http://data.astro.qq.com/dayastro/70/70503/index.shtml";
var astroContent = "";
QQ.LoadScript("http://app.data.qq.com/?umod=astro&act=astro&t=18&jsonp=1&func=qqastro", function(){
});
function qqastro(obj){
    if(QQ.localData.get("astroCustom")){
        astroArr = QQ.localData.get("astroCustom");
    }else{
        if(nowTimeMonth == 3 && nowTimeDay >= 21 || nowTimeMonth == 4 && nowTimeDay <= 19){
            astroToday = "白羊座";
            astroArr = 0;
        }else if(nowTimeMonth == 4 && nowTimeDay >= 20 || nowTimeMonth == 5 && nowTimeDay <= 20){
            astroToday = "金牛座";
            astroArr = 1;
        }else if(nowTimeMonth == 5 && nowTimeDay >= 21 || nowTimeMonth == 6 && nowTimeDay <= 21){
            astroToday = "双子座";
            astroArr = 2;
        }else if(nowTimeMonth == 6 && nowTimeDay >= 22 || nowTimeMonth == 7 && nowTimeDay <= 22){
            astroToday = "巨蟹座";
            astroArr = 3;
        }else if(nowTimeMonth == 7 && nowTimeDay >= 23 || nowTimeMonth == 8 && nowTimeDay <= 22){
            astroToday = "狮子座";
            astroArr = 4;
        }else if(nowTimeMonth == 8 && nowTimeDay >= 23 || nowTimeMonth == 9 && nowTimeDay <= 22){
            astroToday = "处女座";
            astroArr = 5;
        }else if(nowTimeMonth == 9 && nowTimeDay >= 23 || nowTimeMonth == 10 && nowTimeDay <= 23){
            astroToday = "天秤座";
            astroArr = 6;
        }else if(nowTimeMonth == 10 && nowTimeDay >= 24 || nowTimeMonth == 11 && nowTimeDay <= 22){
            astroToday = "天蝎座";
            astroArr = 7;
        }else if(nowTimeMonth == 11 && nowTimeDay >= 23 || nowTimeMonth == 12 && nowTimeDay <= 21){
            astroToday = "射手座";
            astroArr = 8;
        }else if(nowTimeMonth == 12 && nowTimeDay >= 22 || nowTimeMonth == 1 && nowTimeDay <= 19){
            astroToday = "摩羯座";
            astroArr = 9;
        }else if(nowTimeMonth == 1 && nowTimeDay >= 20 || nowTimeMonth == 2 && nowTimeDay <= 18){
            astroToday = "水瓶座";
            astroArr = 10;
        }else if(nowTimeMonth == 2 && nowTimeDay >= 19 || nowTimeMonth == 3 && nowTimeDay <= 20){
            astroToday = "双鱼座";
            astroArr = 11;
        }
    }
    astroYunshi = Number(obj[astroArr].fortune[0].content.replace("%",""));
    astroUrl = "http://data.astro.qq.com"+obj[astroArr].url;
    astroContent = obj[astroArr].fortune[8].content.substring(0, 45)+'...<a href="http://data.astro.qq.com'+obj[astroArr].url+'" target="_blank">[详细]</a>';
    $(".rightXingzuoyunshi .xingzuoSelect").children().eq(astroArr).attr("selected","selected");
    $("#astroIcon").attr("href",astroUrl);
    astroPosition = -(astroArr*50) + "px";
    $("#astroIcon").css("background-position","left "+astroPosition);
    if(astroYunshi >= 95){
        $(".yunshi").width(80);
    }else if(astroYunshi >= 85){
        $(".yunshi").width(72);
    }else if(astroYunshi >= 75){
        $(".yunshi").width(64);
    }else if(astroYunshi >= 65){
        $(".yunshi").width(56);
    }else if(astroYunshi >= 55){
        $(".yunshi").width(48);
    }else if(astroYunshi >= 45){
        $(".yunshi").width(40);
    }else if(astroYunshi >= 35){
        $(".yunshi").width(32);
    }else if(astroYunshi >= 25){
        $(".yunshi").width(24);
    }else if(astroYunshi >= 15){
        $(".yunshi").width(16);
    }else if(astroYunshi >= 5){
        $(".yunshi").width(8);
    }else if(astroYunshi >= 0){
        $(".yunshi").width(0);
    }
    $(".rightXingzuoyunshi .ft p").html(obj[astroArr].fortune[8].content.substring(0, 45)+'...<a href="http://data.astro.qq.com'+obj[astroArr].url+'" target="_blank">[详细]</a>');
    /*添加下拉框事件*/
    $(".rightXingzuoyunshi .xingzuoSelect").bind("change",function(){
        astroYunshi = Number(obj[$(this).val()].fortune[0].content.replace("%",""));
        if(astroYunshi >= 95){
            $(".yunshi").width(80);
        }else if(astroYunshi >= 85){
            $(".yunshi").width(72);
        }else if(astroYunshi >= 75){
            $(".yunshi").width(64);
        }else if(astroYunshi >= 65){
            $(".yunshi").width(56);
        }else if(astroYunshi >= 55){
            $(".yunshi").width(48);
        }else if(astroYunshi >= 45){
            $(".yunshi").width(40);
        }else if(astroYunshi >= 35){
            $(".yunshi").width(32);
        }else if(astroYunshi >= 25){
            $(".yunshi").width(24);
        }else if(astroYunshi >= 15){
            $(".yunshi").width(16);
        }else if(astroYunshi >= 5){
            $(".yunshi").width(8);
        }else if(astroYunshi >= 0){
            $(".yunshi").width(0);
        }
        astroUrl = "http://data.astro.qq.com"+obj[$(this).val()].url;
        $("#astroIcon").attr("href",astroUrl);
        astroPosition = -($(this).val()*50) + "px";
        $("#astroIcon").css("background-position","left "+astroPosition);
        $(".rightXingzuoyunshi .ft p").html(obj[$(this).val()].fortune[8].content.substring(0, 45)+'...<a href="http://data.astro.qq.com'+obj[$(this).val()].url+'" target="_blank">[详细]</a>');
        /*记录选中星座*/
        QQ.localData.set("astroCustom", $(this).val());
    });
}

var userInfoName = "";
/*一键登录数据*/
function loginAll(obj){
    if(obj.result == 0){
        $("#loginedGray").show();
        $("#loginGray").hide();
        MblogTotalnum = obj.info.Mblog.totalnum;
        MblogMsgnum = obj.info.Mblog.msgnum;
        MblogAtnum = obj.info.Mblog.atnum;
        MblogFansnum = obj.info.Mblog.fansnum;

        QZoneTotalnum = obj.info.QZone.totalnum;
        QZonePassivenum = obj.info.QZone.passivenum;
        QZoneInitnum = obj.info.QZone.initnum;
        QZoneAboutnum = obj.info.QZone.aboutnum;

        QMailTotalnum = obj.info.QMail.totalnum;
        QMailInboxnum = obj.info.QMail.inboxnum;
        QMailBottlenum = obj.info.QMail.bottlenum;
        QMailGmailnum = obj.info.QMail.gmailnum;
        QMailDmailnum = obj.info.QMail.dmailnum;

        //DingyueTotalnum = obj.info.Article;

        userInfoName = obj.nick.replace(/</,"&lt;").replace(/>/,"&gt;");
        $("#newsMoreBtn a").html("您有未读新闻，点击查看");
        QQ.localData.set("nick", userInfoName);
        userInfoFace = obj.Face;
        userInfoVip = obj.Vip;

        if(MblogTotalnum){
            if(MblogTotalnum > 99){
                $("#weiboGrayNumLayout").html("99");
            }else{
                $("#weiboGrayNumLayout").html(MblogTotalnum);
            }
            $("#weiboGrayNum").show();
            if(MblogMsgnum != ""){
                $("#msgGrayNum").html(MblogMsgnum);
                $("#msgGrayLi").show();
            }
            if(MblogAtnum != ""){
                $("#atGrayNum").html(MblogAtnum);
                $("#atGrayLi").show();
            }
            if(MblogFansnum != ""){
                $("#fansGrayNum").html(MblogFansnum);
                $("#fansGrayLi").show();
            }
        }else{
            $("#weiboGrayTitle").html("<a href='http://t.qq.com' target='_blank'>点击查看腾讯微博</a>");
            $("#weiboGrayTitle").css("border:0");
        }
        if(QZoneTotalnum){
            if(QZoneTotalnum > 99){
                $("#qzoneGrayNumLayout").html("99");
            }else{
                $("#qzoneGrayNumLayout").html(QZoneTotalnum);
            }
            $("#qzoneGrayNum").show();
            if(QZonePassivenum != ""){
                $("#passiveGrayNum").html(QZonePassivenum);
                $("#passiveGrayLi").show();
            }
            if(QZoneInitnum != ""){
                $("#InitGrayNum").html(QZoneInitnum);
                $("#InitGrayLi").show();
            }
            if(QZoneAboutnum != ""){
                $("#AboutGrayNum").html(QZoneAboutnum);
                $("#AboutGrayLi").show();
            }
        }else{
            $("#qzoneGrayTitle").html("<a href='http://qzone.qq.com' target='_blank'>点击查看QQ空间</a>");
            $("#qzoneGrayTitle").css("border:0");
        }
        if(QMailTotalnum){
            if(QMailTotalnum > 99){
                $("#mailGrayNumLayout").html("99");
            }else{
                $("#mailGrayNumLayout").html(QMailTotalnum);
            }
            $("#mailGrayNum").show();
            weiDu = QMailTotalnum - QMailInboxnum - QMailBottlenum - QMailGmailnum - QMailDmailnum;
            if(QMailInboxnum != ""){
                if(weiDu != 0){
                    $("#inboxGrayNum").html(Number(QMailInboxnum) + Number(weiDu));
                    $("#inboxGrayLi").show();
                }else{
                    $("#inboxGrayNum").html(QMailInboxnum);
                    $("#inboxGrayLi").show();
                }
            }else if(weiDu != 0){
                $("#inboxGrayNum").html(weiDu);
                $("#inboxGrayLi").show();
            }
            if(QMailBottlenum != ""){
                $("#bottleGrayNum").html(QMailBottlenum);
                $("#bottleGrayLi").show();
            }
            if(QMailGmailnum != ""){
                $("#gmailGrayNum").html(QMailGmailnum);
                $("#gmailGrayLi").show();
            }
            if(QMailDmailnum != ""){
                $("#gmailGrayNum").html(QMailDmailnum);
                $("#dmailGrayLi").show();
            }
            if(QMailInboxnum == "" && QMailBottlenum == "" && QMailGmailnum == "" && QMailDmailnum == ""){
                $("#mailGrayTitle").html("<a href='http://mail.qq.com' target='_blank'>点击查看QQ邮箱</a>");
                $("#mailGrayTitle").css("border:0");
            }
        }else{
            $("#mailGrayTitle").html("<a href='http://mail.qq.com' target='_blank'>点击查看QQ邮箱</a>");
            $("#mailGrayTitle").css("border:0");
        }
        /*
         if(DingyueTotalnum){
         $("#dingyueGrayLi").show();
         if(DingyueTotalnum > 99){
         //$("#dingyueGrayNumLayout").html("99");
         $("#dingyueGrayLiNum").html(DingyueTotalnum);
         }else{
         //$("#dingyueGrayNumLayout").html(DingyueTotalnum);
         $("#dingyueGrayLiNum").html(DingyueTotalnum);
         }
         //$("#dingyueGrayNum").show();
         }else{
         $("#dingyueGrayTitle").html("<a href='http://dy.qq.com' target='_blank'>点击进入我的订阅</a>");
         $("#dingyueGrayTitle").css("border:0");
         }
         */
        $("#userNameGray").html("你好，"+userInfoName);
        $("#loginGrayLayoutImg").attr("src",userInfoFace);
        if(userInfoVip){
            $("#userVipLayout").html("VIP"+userInfoVip);
            $("#userVipLayout").show();
            $("#userVipHead").show();
        }
        $("#loginGrayOut").bind("click",function(){
            setTimeout(function(){
                login.loginOut();
            },5);
        });
        //添加图标的mouseover事件
        $("#loginGrayIconLogin").bind("mouseover",function(){
            $("#loginGraySmart").show();
        });
        $("#loginGrayIconLogin").bind("mouseout",function(){
            $("#loginGraySmart").hide();
        });
        $("#dingyueGrayIconLogin").bind("mouseover",function(){
            $("#dingyueGraySmart").show();
        });
        $("#dingyueGrayIconLogin").bind("mouseout",function(){
            $("#dingyueGraySmart").hide();
        });
        $("#mailGrayIconLogin").bind("mouseover",function(){
            $("#mailGraySmart").show();
        });
        $("#mailGrayIconLogin").bind("mouseout",function(){
            $("#mailGraySmart").hide();
        });
        $("#qzoneGrayIconLogin").bind("mouseover",function(){
            $("#qzoneGraySmart").show();
        });
        $("#qzoneGrayIconLogin").bind("mouseout",function(){
            $("#qzoneGraySmart").hide();
        });
        $("#weiboGrayIconLogin").bind("mouseover",function(){
            $("#weiboGraySmart").show();
        });
        $("#weiboGrayIconLogin").bind("mouseout",function(){
            $("#weiboGraySmart").hide();
        });
        QQ.LoadScript("http://i.match.qq.com/dy/mynewsnum?callback=articleNum&ran="+Math.random(),function(){}); //订阅数据接口
        QQ.LoadScript("http://mat1.gtimg.com/finance/js/st/p/qq/q_fin_v20140623144932.js?ran="+Math.random(),function(){
            __.app.qindex.pStock.init();
        });
        ipMe.init();
    }else{
      QQ.LoadScript("http://mat1.gtimg.com/finance/js/st/p/qq/q_fin_v20140623144932.js?ran="+Math.random(),function(){
            __.app.qindex.pStock.init();
        });
        ipMe.init();
        $("#newsMoreBtn a").html("您有未读新闻，点击查看");
        /*
         if(QQ.localData.get("nick")){
         }*/
    }
}
function ilikeFunc(obj){
    if(obj.code == 0){
        if(obj.data.today.length != 0 || obj.data.week.length != 0 || obj.data.early.length != 0){
            if(obj.data.today.length != 0){
                var iliketodayArr = [];
                for(var i=0;i<obj.data.today.length;i++){
                    if(obj.data.today[i].lt){
                        iliketodayArr.push('<dd><a target="_blank" href="http://' + obj.data.today[i].url + '">' + obj.data.today[i].lt + '</a></dd>');
                    }
                }
                $("#iliketoday").html('<dt>今日</dt>' + iliketodayArr.join(""));
                $("#iliketoday").show();
                $("#ilikeSmartNo").attr("id","ilikeSmart");
                $("#ilikeTitle").css("text-indent:0px");
            }
            if(obj.data.week.length != 0){
                var ilikeweekArr = [];
                for(var i=0;i<obj.data.week.length;i++){
                    if(obj.data.week[i].lt){
                        ilikeweekArr.push('<dd><a target="_blank" href="http://' + obj.data.week[i].url + '">' + obj.data.week[i].lt + '</a></dd>');
                    }
                }
                $("#ilikeweek").html('<dt>本周</dt>' + ilikeweekArr.join(""));
                $("#ilikeweek").show();
                $("#ilikeSmartNo").attr("id","ilikeSmart");
                $("#ilikeTitle").css("text-indent:0px");
            }
            if(obj.data.early.length != 0){
                var ilikeearlyArr = [];
                for(var i=0;i<obj.data.early.length;i++){
                    if(obj.data.early[i].lt){
                        ilikeearlyArr.push('<dd><a target="_blank" href="http://' + obj.data.early[i].url + '">' + obj.data.early[i].lt + '</a></dd>');
                    }
                }
                $("#ilikeearly").html('<dt>更早</dt>' + ilikeearlyArr.join(""));
                $("#ilikeearly").show();
                $("#ilikeSmartNo").attr("id","ilikeSmart");
                $("#ilikeTitle").css("text-indent:0px");
            }
            $("#ilikeTitle").html('最近收藏:');
            $("#ilikeContent").show();
        }
        $("#ilikeIcon").bind("mouseover",function(){
            $("#ilikeFloat").show();
        });
        $("#ilikeIcon").bind("mouseout",function(){
            $("#ilikeFloat").hide();
        });
    }
}
function articleNum(obj){
    if(obj.code == 0){
        DingyueTotalnum = obj.data.total;
        if(DingyueTotalnum){
            $("#dingyueGrayLi").show();
            if(DingyueTotalnum > 99){
                $("#dingyueGrayLiNum").html(DingyueTotalnum);
            }else{
                $("#dingyueGrayLiNum").html(DingyueTotalnum);
            }
        }else{
            $("#dingyueGrayTitle").html("<a href='http://dy.qq.com' target='_blank'>点击进入我的订阅</a>");
            $("#dingyueGrayTitle").css("border:0");
        }
    }else{
        $("#dingyueGrayTitle").html("<a href='http://dy.qq.com' target='_blank'>点击进入我的订阅</a>");
        $("#dingyueGrayTitle").css("border:0");
        $("#dingyueGrayLiNum").hide();
    }
}

function getExpires(a){//a:hour
    var expires=new Date(new Date().getTime()+(a?a:1)*3600*1000);
    return expires;
}

var nowTimeYear = serverTime.getFullYear();
var nowTimeMonth = serverTime.getMonth() + 1;
var nowTimeDay = serverTime.getDate();
var nowTimeWeek = serverTime.getDay();
var nowTimeHour = serverTime.getHours();
var nowTimeMinute = serverTime.getMinutes();
var nowTimeSecond = serverTime.getSeconds();
var nowWeek = ["星期日","星期一","星期二","星期三","星期四","星期五","星期六"];

var navMap ={
    "上海市":"navShanghai",
    "重庆市":"navChongqing",
    "广东省":"navGuangdong",
  	"陕西省":"navShanxi",
    "四川省":"navSichuan",
    "福建省":"navFujian",
    "河南省":"navHenan",
    "湖北省":"navHubei",
    "湖南省":"navHunan",
    "辽宁省":"navLiaoning",
    "江苏省":"navJiangsu",
    "浙江省":"navZhejiang"
}
var newsMap ={
    "上海市":"<a href='http://sh.qq.com/' bossZone='shnewstab' target='_blank'>上海新闻</a>",
    "重庆市":"<a href='http://cq.qq.com/' bossZone='cqnewstab' target='_blank'>重庆新闻</a>",
    "广东省":"<a href='http://gd.qq.com/' bossZone='gdnewstab' target='_blank'>广东新闻</a>",
  	"陕西省":"<a href='http://xian.qq.com/' bossZone='xiannewstab' target='_blank'>陕西新闻</a>",
    "四川省":"<a href='http://cd.qq.com/' bossZone='cdnewstab' target='_blank'>四川新闻</a>",
    "福建省":"<a href='http://fj.qq.com/' bossZone='fjnewstab' target='_blank'>福建新闻</a>",
    "河南省":"<a href='http://henan.qq.com/' bossZone='henannewstab' target='_blank'>河南新闻</a>",
    "湖北省":"<a href='http://hb.qq.com/' bossZone='hbnewstab' target='_blank'>湖北新闻</a>",
    "湖南省":"<a href='http://hn.qq.com/' bossZone='hunannewstab' target='_blank'>湖南新闻</a>",
    "辽宁省":"<a href='http://ln.qq.com/' bossZone='lnnewstab' target='_blank'>辽宁新闻</a>",
    "江苏省":"<a href='http://js.qq.com/' bossZone='jsnewstab' target='_blank'>江苏新闻</a>",
    "浙江省":"<a href='http://zj.qq.com/' bossZone='zjnewstab' target='_blank'>浙江新闻</a>"
}
var newsInfoMap ={
    "上海市":"/ninja/newsShanghaiContent_public.htm",
    "重庆市":"/ninja/newsChongqingContent_public.htm",
    "广东省":"/ninja/newsGuangdongContent_public.htm",
  	"陕西省":"/ninja/newsShanxiContent_public.htm",
    "四川省":"/ninja/newsSichuanContent_public.htm",
    "福建省":"/ninja/newsFujianContent_public.htm",
    "河南省":"/ninja/newsHenanContent_public.htm",
    "湖北省":"/ninja/newsHubeiContent_public.htm",
    "湖南省":"/ninja/newsHunanContent_public.htm",
    "辽宁省":"/ninja/newsLiaoningContent_public.htm",
    "江苏省":"/ninja/newsJiangsuContent_public.htm",
    "浙江省":"/ninja/newsZhejiangContent_public.htm"
}
var tuanDifangMap ={
    "上海市":"/v/sh/www_shanghai_pic.htm",
    "重庆市":"/v/cq/www_chongqing_pic.htm",
    "广东省":"/v/gd/www_guangdong_pic.htm",
  	"陕西省":"/v/xian/www_shanxi_pic.htm",
    "四川省":"/v/cd/www_sichuan_pic.htm",
    "福建省":"/v/fj/www_fujian_pic.htm",
    "河南省":"/v/henan/www_henan_pic.htm",
    "湖北省":"/v/hb/www_hubei_pic.htm",
    "湖南省":"/v/hn/www_hunan_pic.htm",
    "辽宁省":"/v/ln/www_liaoning_pic.htm",
    "江苏省":"/v/jiangsu/www_jiangsu_pic.htm",
    "浙江省":"/v/zj/www_zhejiang_pic.htm"
}
var tuanMap ={
    "合肥市":"合肥市",
    //"福州市":"福州市",
    //"厦门市":"厦门市",
    //"东莞市":"东莞市",
    //"广州市":"广州市",
    //"深圳市":"深圳市",
    "桂林市":"桂林市",
    "兰州市":"兰州市",
    "贵阳市":"贵阳市",
    "保定市":"保定市",
    //"洛阳市":"洛阳市",
    //"郑州市":"郑州市",
    "海口市":"海口市",
    "三亚市":"三亚市",
    //"武汉市":"武汉市",
    //"长沙市":"长沙市",
    "哈尔滨市":"哈尔滨市",
    "长春市":"长春市",
    "常州市":"常州市",
    "南京市":"南京市",
    "南昌市":"南昌市",
    //"大连市":"大连市",
    //"沈阳市":"沈阳市",
    "呼和浩特市":"呼和浩特市",
    "银川市":"银川市",
    "西宁市":"西宁市",
    //"成都市":"成都市",
    "济南市":"济南市",
    "青岛市":"青岛市",
    "大同市":"大同市",
    "太原市":"太原市",
    //"西安市":"西安市",
    "台北":"台北",
    "乌鲁木齐市":"乌鲁木齐市",
    "拉萨市":"拉萨市",
    "昆明市":"昆明市",
    "石家庄市":"石家庄市",
    "南宁市":"南宁市",
    "丽江市":"丽江市",
    "苏州市":"苏州市",
    "香港":"香港",
    "澳门":"澳门",
    "北京市":"北京市",
    //"杭州市":"杭州市",
    //"温州市":"温州市",
    //"重庆市":"重庆市",
    //"上海市":"上海市"
    "天津市":"天津市"
}
var tuanTitleMap ={
    "上海市":"<a href='http://sh.qq.com/' bossZone='shlifeTAB' target='_blank'>上海</a>·<a href='http://mysh.qq.com/forum.php' bossZone='shlifeTAB' target='_blank'>生活</a>",
    "重庆市":"<a href='http://cq.qq.com/' bossZone='cqlifeTAB' target='_blank'>重庆</a>·<a href='http://mycq.qq.com/' bossZone='cqlifeTAB' target='_blank'>生活</a>",
    "广东省":"<a href='http://gd.qq.com/' bossZone='gdlifeTAB' target='_blank'>广东</a>·<a href='http://mygd.qq.com/' bossZone='gdlifeTAB' target='_blank'>生活</a>",
  	"陕西省":"<a href='http://xian.qq.com/' bossZone='xianlifeTAB' target='_blank'>陕西</a>·<a href='http://myxian.qq.com/forum.php' bossZone='xianlifeTAB' target='_blank'>生活</a>",
    "四川省":"<a href='http://cd.qq.com/' bossZone='cdlifeTAB' target='_blank'>四川</a>·<a href='http://mycd.qq.com/' bossZone='cdlifeTAB' target='_blank'>生活</a>",
    "福建省":"<a href='http://fj.qq.com/' bossZone='fjlifeTAB' target='_blank'>福建</a>·<a href='http://myfj.qq.com/' bossZone='fjlifeTAB' target='_blank'>生活</a>",
    "河南省":"<a href='http://henan.qq.com/' bossZone='henanlifeTAB' target='_blank'>河南</a>·<a href='http://myhenan.qq.com/' bossZone='henanlifeTAB' target='_blank'>生活</a>",
    "湖北省":"<a href='http://hb.qq.com/' bossZone='hblifeTAB' target='_blank'>湖北</a>·<a href='http://myhb.qq.com/forum.php' bossZone='hblifeTAB' target='_blank'>生活</a>",
    "湖南省":"<a href='http://hn.qq.com/' bossZone='hunanlifeTAB' target='_blank'>湖南</a>·<a href='http://myhn.qq.com/forum.php' bossZone='hunanlifeTAB' target='_blank'>生活</a>",
    "辽宁省":"<a href='http://ln.qq.com/' bossZone='lnlifeTAB' target='_blank'>辽宁</a>·<a href='http://myln.qq.com/forum.php' bossZone='lnlifeTAB' target='_blank'>生活</a>",
    "江苏省":"<a href='http://js.qq.com/' bossZone='jslifeTAB' target='_blank'>江苏</a>·<a href='http://myjs.qq.com/' bossZone='jslifeTAB' target='_blank'>生活</a>",
    "浙江省":"<a href='http://zj.qq.com/' bossZone='zjlifeTAB' target='_blank'>浙江</a>·<a href='http://myzj.qq.com/' bossZone='zjlifeTAB' target='_blank'>生活</a>",
    "山西省":"<a href='http://vip.qq.com/life.html' bossZone='shanxilifeTAB' target='_blank'>山西生活</a>",
    "安徽省":"<a href='http://vip.qq.com/life.html' bossZone='ahlifeTAB' target='_blank'>安徽生活</a>",
    "贵州省":"<a href='http://vip.qq.com/life.html' bossZone='gzlifeTAB' target='_blank'>贵州生活</a>",
    "云南省":"<a href='http://vip.qq.com/life.html' bossZone='ynlifeTAB' target='_blank'>云南生活</a>",
    "吉林省":"<a href='http://vip.qq.com/life.html' bossZone='jllifeTAB' target='_blank'>吉林生活</a>",
    "天津市":"<a target='_blank' href='http://e.t.qq.com/TJB1009' bossZone='tjlifeTAB'>天津生活</a>"
}
var tuanTxtMap ={
    "上海市":"/v/sh/www_shanghai_txt.htm",
    "重庆市":"/v/cq/www_chongqing_txt.htm",
    "广东省":"/v/gd/www_guangdong_txt.htm",
  	"陕西省":"/v/xian/www_shanxi_txt.htm",
    "四川省":"/v/cd/www_sichuan_txt.htm",
    "福建省":"/v/fj/www_fujian_txt.htm",
    "河南省":"/v/henan/www_henan_txt.htm",
    "湖北省":"/v/hb/www_hubei_txt.htm",
    "湖南省":"/v/hn/www_hunan_txt.htm",
    "辽宁省":"/v/ln/www_liaoning_txt.htm",
    "江苏省":"/v/jiangsu/www_jiangsu_txt.htm",
    "浙江省":"/v/zj/www_zhejiang_txt.htm"
}
var liveMap ={
    "上海市":"/ninja/liveShanghai_NOINCLUDE_WITHOUTPULL.htm",
    "重庆市":"/ninja/liveChongqing_NOINCLUDE_WITHOUTPULL.htm",
    "广东省":"/ninja/liveGuangdong_NOINCLUDE_WITHOUTPULL.htm",
  	"陕西省":"/ninja/liveShanxi_NOINCLUDE_WITHOUTPULL.htm",
    "四川省":"/ninja/liveSichuan_NOINCLUDE_WITHOUTPULL.htm",
    "福建省":"/ninja/liveFujian_NOINCLUDE_WITHOUTPULL.htm",
    "河南省":"/ninja/liveHenan_NOINCLUDE_WITHOUTPULL.htm",
    "湖北省":"/ninja/liveHubei_NOINCLUDE_WITHOUTPULL.htm",
    "湖南省":"/ninja/liveHunan_NOINCLUDE_WITHOUTPULL.htm",
    "辽宁省":"/ninja/liveLiaoning_NOINCLUDE_WITHOUTPULL.htm",
    "江苏省":"/ninja/liveJiangsu_NOINCLUDE_WITHOUTPULL.htm",
    "浙江省":"/ninja/liveZhejiang_NOINCLUDE_WITHOUTPULL.htm",
    "山西省":"/ninja/liveTaiyuan_NOINCLUDE_WITHOUTPULL.htm",
    "安徽省":"/ninja/liveAnhui_NOINCLUDE_WITHOUTPULL.htm",
    "贵州省":"/ninja/liveGuizhou_NOINCLUDE_WITHOUTPULL.htm",
    "吉林省":"/ninja/liveJilin_NOINCLUDE_WITHOUTPULL.htm",
    "云南省":"/ninja/liveYunnan_NOINCLUDE_WITHOUTPULL.htm",
    "天津市":"/ninja/liveTianjin_NOINCLUDE_WITHOUTPULL.htm"
}
var meishiMap ={
    "上海市":"/ninja/meishiShanghai_NOINCLUDE_WITHOUTPULL.htm",
    "重庆市":"/ninja/meishiChongqing_NOINCLUDE_WITHOUTPULL.htm",
    "广东省":"/ninja/meishiGuangdong_NOINCLUDE_WITHOUTPULL.htm",
  	"陕西省":"/ninja/meishiShanxi_NOINCLUDE_WITHOUTPULL.htm",
    "四川省":"/ninja/meishiSichuan_NOINCLUDE_WITHOUTPULL.htm",
    "福建省":"/ninja/meishiFujian_NOINCLUDE_WITHOUTPULL.htm",
    "河南省":"/ninja/meishiHenan_NOINCLUDE_WITHOUTPULL.htm",
    "湖北省":"/ninja/meishiHubei_NOINCLUDE_WITHOUTPULL.htm",
    "湖南省":"/ninja/meishiHunan_NOINCLUDE_WITHOUTPULL.htm",
    "辽宁省":"/ninja/meishiLiaoning_NOINCLUDE_WITHOUTPULL.htm",
    "江苏省":"/ninja/meishiJiangsu_NOINCLUDE_WITHOUTPULL.htm",
    "浙江省":"/ninja/meishiZhejiang_NOINCLUDE_WITHOUTPULL.htm"
}
var houseTitleMap ={
    "北京市":"<a href='http://bj.house.qq.com/' bossZone='bjhousetab' target='_blank'>北京房产</a>",
    "上海市":"<a href='http://sh.house.qq.com/' bossZone='shhousetab' target='_blank'>上海房产</a>",
    "四川省":"<a href='http://cd.house.qq.com/' bossZone='cdhousetab' target='_blank'>成都房产</a>",
    "重庆市":"<a href='http://cq.house.qq.com/' bossZone='cqhousetab' target='_blank'>重庆房产</a>",
    "福建省":"<a href='http://fz.house.qq.com/' bossZone='fzhousetab' target='_blank'>福州房产</a>",
    "厦门市":"<a href='http://xm.house.qq.com/' bossZone='xmhousetab' target='_blank'>厦门房产</a>",
    "广东省":"<a href='http://gz.house.qq.com/' bossZone='gzhousetab' target='_blank'>广州房产</a>",
    "深圳市":"<a href='http://sz.house.qq.com/' bossZone='szhousetab' target='_blank'>深圳房产</a>",
    "珠海市":"<a href='http://zh.house.qq.com/' bossZone='zhhousetab' target='_blank'>珠海房产</a>",
    "河南省":"<a href='http://zz.house.qq.com/' bossZone='zzhousetab' target='_blank'>郑州房产</a>",
    "湖北省":"<a href='http://wh.house.qq.com/' bossZone='whhousetab' target='_blank'>武汉房产</a>",
    "湖南省":"<a href='http://cs.house.qq.com/' bossZone='cshousetab' target='_blank'>长沙房产</a>",
  	"陕西省":"<a href='http://xian.house.qq.com/' bossZone='xianhousetab' target='_blank'>西安房产</a>",
    "辽宁省":"<a href='http://sy.house.qq.com/' bossZone='syhousetab' target='_blank'>沈阳房产</a>",
    "江苏省":"<a href='http://nj.house.qq.com/' bossZone='njhousetab' target='_blank'>南京房产</a>",
    "杭州市":"<a href='http://hz.house.qq.com/' bossZone='hzhousetab' target='_blank'>杭州房产</a>",
    "佛山市":"<a href='http://fs.house.qq.com/' bossZone='fshousetab' target='_blank'>佛山房产</a>",
    "惠州市":"<a href='http://huizhou.house.qq.com/' bossZone='huizhouhousetab' target='_blank'>惠州房产</a>",
    "东莞市":"<a href='http://dg.house.qq.com/' bossZone='dghousetab' target='_blank'>东莞房产</a>",
    "中山市":"<a href='http://zh.house.qq.com/' bossZone='zshousetab' target='_blank'>中山房产</a>",
    "江门市":"<a href='http://zh.house.qq.com/jm.htm' bossZone='jmhousetab' target='_blank'>江门房产</a>"
}
var houseMap ={
    "北京市":"/ninja/houseBeijingContent_public.htm",
    "上海市":"/ninja/houseShanghaiContent_public.htm",
    "四川省":"/ninja/houseChengduContent_public.htm",
    "重庆市":"/ninja/houseChongqingContent_public.htm",
    "福建省":"/ninja/houseFuzhouContent_public.htm",
    "厦门市":"/ninja/houseXiamenContent_public.htm",
    "广东省":"/ninja/houseGuangzhouContent_public.htm",
    "深圳市":"/ninja/houseShenzhenContent_public.htm",
    "珠海市":"/ninja/houseZhuhaiContent_public.htm",
    "河南省":"/ninja/houseZhengzhouContent_public.htm",
    "湖北省":"/ninja/houseWuhanContent_public.htm",
    "湖南省":"/ninja/houseChangshaContent_public.htm",
  	"陕西省":"/ninja/houseXianContent_public.htm",
    "辽宁省":"/ninja/houseShenyangContent_public.htm",
    "江苏省":"/ninja/houseNanjingContent_public.htm",
    "杭州市":"/ninja/houseHangzhouContent_public.htm",
    "佛山市":"/ninja/houseFoshanContent_public.htm",
    "惠州市":"/ninja/houseHuizhouContent_public.htm",
    "东莞市":"/ninja/houseDongguanContent_public.htm",
    "中山市":"/ninja/houseZhongshanContent_public.htm",
    "江门市":"/ninja/houseJiangmenContent_public.htm"
    /*
     "浙江省":"/v/househangzhou/www_hangzhou_house.htm",
     "莆田市":"/v/house_fuzhou/www_fuzhou_house.htm",
     "宁德市":"/v/house_fuzhou/www_fuzhou_house.htm",
     "南平市":"/v/house_fuzhou/www_fuzhou_house.htm",
     "三明市":"/v/house_fuzhou/www_fuzhou_house.htm",
     "漳州市":"/v/house_xiamen/www_xiamen_house.htm",
     "泉州市":"/v/house_xiamen/www_xiamen_house.htm",
     "龙岩市":"/v/house_xiamen/www_xiamen_house.htm",
     "惠州市":"/v/house_shenzhen/www_shenzhen_house.htm",
     "东莞市":"/v/house_shenzhen/www_shenzhen_house.htm",
     "中山市":"/v/house_zhuhai/www_zhuhai_house.htm",
     "江门市":"/v/house_zhuhai/www_zhuhai_house.htm",
     */
}
var autoMap ={
    "广东省":"/v/auto/www_auto_gd.htm",
    "湖北省":"/v/auto/www_auto_hb.htm",
    "重庆市":"/v/auto/www_auto_cq.htm"
}
var autoPriceMap ={
    "北京市":"北京市",
    "上海市":"上海市",
    "重庆市":"重庆市",
    "西安市":"西安市",
    "武汉市":"武汉市",
    "长沙市":"长沙市",
    "南京市":"南京市",
    "沈阳市":"沈阳市",
    "郑州市":"郑州市",
    "成都市":"成都市",
    "杭州市":"杭州市",
    "宁波市":"宁波市",
    "广州市":"广州市",
    "深圳市":"深圳市",
    "东莞市":"东莞市",
    "佛山市":"佛山市",
    "中山市":"中山市",
    "福州市":"福州市",
    "泉州市":"泉州市",
    "济南市":"济南市",
    "青岛市":"青岛市",
    "南昌市":"南昌市",
    "合肥市":"合肥市"
}
var difangPuguangMap ={
    "上海市":"shexp",
    "重庆市":"cqexp",
    "广东省":"gdexp",
  	"陕西省":"xianexp",
    "四川省":"cdexp",
    "福建省":"fjexp",
    "河南省":"henanexp",
    "湖北省":"hbexp",
    "湖南省":"hunanexp",
    "辽宁省":"lnexp",
    "江苏省":"jsexp",
    "浙江省":"zjexp"
}
var difangTabMap ={
    "上海市":"shtexp",
    "重庆市":"cqtexp",
    "广东省":"gdtexp",
  	"陕西省":"xiantexp",
    "四川省":"cdtexp",
    "福建省":"fjtexp",
    "河南省":"henantexp",
    "湖北省":"hbtexp",
    "湖南省":"hunantexp",
    "辽宁省":"lntexp",
    "江苏省":"jstexp",
    "浙江省":"zjtexp"
}
var provincialCapitalMap ={
    "安徽省":"合肥市",
    "福建省":"福州市",
    "甘肃省":"兰州市",
    "广东省":"广州市",
    "贵州省":"贵阳市",
    "海南省":"海口市",
    "河北省":"石家庄市",
    "河南省":"郑州市",
    "黑龙江省":"哈尔滨市",
    "湖北省":"武汉市",
    "湖南省":"长沙市",
    "吉林省":"长春市",
    "江苏省":"南京市",
    "江西省":"南昌市",
    "辽宁省":"沈阳市",
    "青海省":"西宁市",
    "山东省":"济南市",
    "山西省":"太原市",
  	"陕西省":"西安市",
    "四川省":"成都市",
    "云南省":"昆明市",
    "浙江省":"杭州市",
    "台湾省":"台北市",
    "广西":"南宁市",
    "内蒙古":"呼和浩特市",
    "西藏":"拉萨市",
    "宁夏":"银川市",
    "新疆":"乌鲁木齐市",
    "北京市":"北京市",
    "上海市":"上海市",
    "重庆市":"重庆市",
    "天津市":"天津市",
    "香港":"香港市",
    "澳门":"澳门市"
}
var cityCode = {
    "北京市":{
        "北京市":{}
    },
    "上海市":{
        "上海市":{}
    },
    "天津市":{
        "天津市":{}
    },
    "重庆市":{
        "重庆市":{}
    },
    "河北省":{
        "石家庄市":{},
        "唐山市":{},
        "秦皇岛市":{},
        "邯郸市":{},
        "邢台市":{},
        "保定市":{},
        "张家口市":{},
        "承德市":{},
        "沧州市":{},
        "廊坊市":{},
        "衡水市":{}
    },
    "山西省":{
        "太原市":{},
        "大同市":{},
        "阳泉市":{},
        "长治市":{},
        "晋城市":{},
        "朔州市":{},
        "晋中市":{},
        "运城市":{},
        "忻州市":{},
        "临汾市":{},
        "吕梁市":{}
    },
    "内蒙古":{
        "呼和浩特市":{},
        "包头市":{},
        "乌海市":{},
        "赤峰市":{},
        "通辽市":{},
        "鄂尔多斯市":{},
        "呼伦贝尔市":{},
        "巴彦淖尔市":{},
        "乌兰察布市":{},
        "锡林郭勒盟":{},
        "阿拉善盟":{},
        "兴安盟":{}
    },
    "江苏省":{
        "南京市":{},
        "无锡市":{},
        "徐州市":{},
        "常州市":{},
        "苏州市":{},
        "南通市":{},
        "连云港市":{},
        "淮安市":{},
        "盐城市":{},
        "扬州市":{},
        "镇江市":{},
        "泰州市":{},
        "宿迁市":{}
    },
    "安徽省":{
        "合肥市":{},
        "芜湖市":{},
        "蚌埠市":{},
        "淮南市":{},
        "马鞍山市":{},
        "淮北市":{},
        "铜陵市":{},
        "安庆市":{},
        "黄山市":{},
        "滁州市":{},
        "阜阳市":{},
        "宿州市":{},
        "巢湖市":{},
        "六安市":{},
        "亳州市":{},
        "池州市":{},
        "宣城市":{}
    },
    "山东省":{
        "济南市":{},
        "青岛市":{},
        "淄博市":{},
        "枣庄市":{},
        "东营市":{},
        "潍坊市":{},
        "烟台市":{},
        "威海市":{},
        "济宁市":{},
        "泰安市":{},
        "日照市":{},
        "莱芜市":{},
        "临沂市":{},
        "德州市":{},
        "聊城市":{},
        "滨州市":{},
        "菏泽市":{}
    },
    "辽宁省":{
        "沈阳市":{},
        "大连市":{},
        "鞍山市":{},
        "抚顺市":{},
        "本溪市":{},
        "丹东市":{},
        "锦州市":{},
        "营口市":{},
        "阜新市":{},
        "辽阳市":{},
        "盘锦市":{},
        "铁岭市":{},
        "朝阳市":{},
        "葫芦岛市":{}
    },
    "吉林省":{
        "长春市":{},
        "吉林市":{},
        "四平市":{},
        "辽源市":{},
        "通化市":{},
        "白山市":{},
        "松原市":{},
        "白城市":{},
        "延边州":{}
    },
    "黑龙江省":{
        "哈尔滨市":{},
        "齐齐哈尔市":{},
        "鸡西市":{},
        "鹤岗市":{},
        "双鸭山市":{},
        "大庆市":{},
        "伊春市":{},
        "牡丹江市":{},
        "佳木斯市":{},
        "七台河市":{},
        "黑河市":{},
        "绥化市":{},
        "大兴安岭地区":{}
    },
    "浙江省":{
        "杭州市":{},
        "宁波市":{},
        "温州市":{},
        "嘉兴市":{},
        "湖州市":{},
        "绍兴市":{},
        "金华市":{},
        "衢州市":{},
        "舟山市":{},
        "台州市":{},
        "丽水市":{}
    },
    "江西省":{
        "南昌市":{},
        "景德镇市":{},
        "萍乡市":{},
        "九江市":{},
        "新余市":{},
        "鹰潭市":{},
        "赣州市":{},
        "吉安市":{},
        "宜春市":{},
        "抚州市":{},
        "上饶市":{}
    },
    "福建省":{
        "福州市":{},
        "厦门市":{},
        "莆田市":{},
        "三明市":{},
        "泉州市":{},
        "漳州市":{},
        "南平市":{},
        "龙岩市":{},
        "宁德市":{}
    },
    "湖北省":{
        "武汉市":{},
        "黄石市":{},
        "襄樊市":{},
        "十堰市":{},
        "荆州市":{},
        "宜昌市":{},
        "荆门市":{},
        "鄂州市":{},
        "孝感市":{},
        "黄冈市":{},
        "咸宁市":{},
        "随州市":{},
        "恩施州":{},
        "仙桃市":{},
        "天门市":{},
        "潜江市":{},
        "神农架林区":{}
    },
    "湖南省":{
        "长沙市":{},
        "株洲市":{},
        "湘潭市":{},
        "衡阳市":{},
        "邵阳市":{},
        "岳阳市":{},
        "常德市":{},
        "张家界市":{},
        "益阳市":{},
        "郴州市":{},
        "永州市":{},
        "怀化市":{},
        "娄底市":{},
        "湘西州":{}
    },
    "河南省":{
        "郑州市":{},
        "开封市":{},
        "洛阳市":{},
        "平顶山市":{},
        "焦作市":{},
        "鹤壁市":{},
        "新乡市":{},
        "安阳市":{},
        "濮阳市":{},
        "许昌市":{},
        "漯河市":{},
        "三门峡市":{},
        "南阳市":{},
        "商丘市":{},
        "信阳市":{},
        "周口市":{},
        "驻马店市":{},
        "济源市":{}
    },
    "广东省":{
        "广州市":{},
        "深圳市":{},
        "珠海市":{},
        "汕头市":{},
        "韶关市":{},
        "河源市":{},
        "梅州市":{},
        "惠州市":{},
        "汕尾市":{},
        "东莞市":{},
        "中山市":{},
        "江门市":{},
        "佛山市":{},
        "阳江市":{},
        "湛江市":{},
        "茂名市":{},
        "肇庆市":{},
        "清远市":{},
        "潮州市":{},
        "揭阳市":{},
        "云浮市":{}
    },
    "广西":{
        "南宁市":{},
        "柳州市":{},
        "桂林市":{},
        "梧州市":{},
        "北海市":{},
        "防城港市":{},
        "钦州市":{},
        "贵港市":{},
        "玉林市":{},
        "百色市":{},
        "贺州市":{},
        "河池市":{},
        "来宾市":{},
        "崇左市":{}
    },
    "海南省":{
        "海口市":{},
        "三亚市":{},
        "五指山市":{},
        "琼海市":{},
        "儋州市":{},
        "文昌市":{},
        "万宁市":{},
        "东方市":{},
        "澄迈县":{},
        "定安县":{},
        "屯昌县":{},
        "临高县":{},
        "白沙黎族自治县":{}
    },
    "四川省":{
        "成都市":{},
        "自贡市":{},
        "攀枝花市":{},
        "泸州市":{},
        "德阳市":{},
        "绵阳市":{},
        "广元市":{},
        "遂宁市":{},
        "内江市":{},
        "乐山市":{},
        "南充市":{},
        "宜宾市":{},
        "广安市":{},
        "达州市":{},
        "眉山市":{},
        "雅安市":{},
        "巴中市":{},
        "资阳市":{},
        "阿坝州":{},
        "甘孜州":{},
        "凉山州":{}
    },
    "贵州省":{
        "贵阳市":{},
        "六盘水市":{},
        "遵义市":{},
        "安顺市":{},
        "铜仁地区":{},
        "毕节地区":{},
        "黔西南州":{},
        "黔东南州":{},
        "黔南州":{}
    },
    "云南省":{
        "昆明市":{},
        "曲靖市":{},
        "玉溪市":{},
        "保山市":{},
        "昭通市":{},
        "丽江市":{},
        "思茅市":{},
        "临沧市":{},
        "文山州":{},
        "红河州":{},
        "西双版纳州":{},
        "楚雄州":{},
        "大理州":{},
        "德宏州":{},
        "怒江州":{},
        "迪庆州":{}
    },
    "西藏":{
        "拉萨市":{},
        "那曲地区":{},
        "昌都地区":{},
        "山南地区":{},
        "日喀则地区":{},
        "阿里地区":{},
        "林芝地区":{}
    },
    "陕西省":{
        "西安市":{},
        "铜川市":{},
        "宝鸡市":{},
        "咸阳市":{},
        "渭南市":{},
        "延安市":{},
        "汉中市":{},
        "榆林市":{},
        "安康市":{},
        "商洛市":{}
    },
    "甘肃省":{
        "兰州市":{},
        "金昌市":{},
        "白银市":{},
        "天水市":{},
        "嘉峪关市":{},
        "武威市":{},
        "张掖市":{},
        "平凉市":{},
        "酒泉市":{},
        "庆阳市":{},
        "定西市":{},
        "陇南市":{},
        "临夏州":{},
        "甘南州":{}
    },
    "宁夏":{
        "银川市":{},
        "石嘴山市":{},
        "吴忠市":{},
        "固原市":{},
        "中卫市":{}
    },
    "青海省":{
        "西宁市":{},
        "海东地区":{},
        "海北州":{},
        "黄南州":{},
        "海南州":{},
        "果洛州":{},
        "玉树州":{},
        "海西州":{}
    },
    "新疆":{
        "乌鲁木齐市":{},
        "克拉玛依市":{},
        "吐鲁番地区":{},
        "哈密地区":{},
        "和田地区":{},
        "阿克苏地区":{},
        "喀什地区":{},
        "克孜勒苏柯州":{},
        "巴音郭楞州":{},
        "昌吉州":{},
        "博尔塔拉州":{},
        "伊犁州":{},
        "塔城地区":{},
        "阿勒泰地区":{},
        "石河子市":{},
        "阿拉尔市":{}
    },
    "香港":{
        "香港":{}
    },
    "澳门":{
        "澳门":{}
    },
    "台湾省":{
        "台北":{},
        "高雄":{},
        "台中":{},
        "花莲":{},
        "基隆":{},
        "嘉义":{},
        "金门":{},
        "连江":{},
        "苗栗":{},
        "南投":{},
        "澎湖":{},
        "屏东":{},
        "台东":{},
        "台南":{},
        "桃园":{},
        "新竹":{},
        "宜兰":{},
        "云林":{},
        "彰化":{}
    }
}

//地方站要闻页卡隐藏推广位
function updateStatus(){
    if( $('#newsH2').css('display') != 'none' || $("#newsQuanguoH2").hasClass('selected') ){
        $("#diaocha").show();
    }else{
        $("#diaocha").hide();
    }
}

var ipMe = {
    pro:'北京市',
    city:'',
    newscheck:function(proName){
        if(newsMap[proName]){
            var ajaxNews = new Ajax({
                headers:{"If-Modified-Since":0},
                url:newsInfoMap[proName],
                method:"get"
            });
            ajaxNews.onComplete = function(responseObject){
                $("#newsDifangH2").html(newsMap[proName]);
                $("#newsDifangH2").addClass("select");
                $("#newsDifangH2").show();
                $("#newsInfo").html(responseObject.textString);
            };
            ajaxNews.start();
            $("#newsH2").hide();
            $("#qq-tabs2").show();
            bossPuguang(difangTabMap[proName],"page");
        }else{
            $("#newsH2").show();
            $("#qq-tabs2").hide();
            $("#newsInfoQuanguo").show();
            $("#newsInfo").hide();
        }
    },
    weathercheck:function(proName,cityName,type){
        QQ.LoadScript("http://mat1.gtimg.com/www/js/qq2012/weatherNew_1.5.js",function(){
            if(weatherMap[proName]){
                if(weatherMap[proName][cityName]){
                    weatherId = weatherMap[proName][cityName];
                }else{
                    weatherId = weatherMap[proName]["_"];
                }
            }else{
                weatherId = weatherMap["defaultCity"];
            }
            var weatherCityMap ={
                "weatherCityWind":{
                    "0":{
                        "Dir":"",
                        "Power":"微风"
                    },
                    "1":{
                        "Dir":"东北风",
                        "Power":"3-4级"
                    },
                    "2":{
                        "Dir":"东风",
                        "Power":"4-5级"
                    },
                    "3":{
                        "Dir":"东南风",
                        "Power":"5-6级"
                    },
                    "4":{
                        "Dir":"南风",
                        "Power":"6-7级"
                    },
                    "5":{
                        "Dir":"西南风",
                        "Power":"7-8级"
                    },
                    "6":{
                        "Dir":"西风",
                        "Power":"8-9级"
                    },
                    "7":{
                        "Dir":"西北风",
                        "Power":"9-10级"
                    },
                    "8":{
                        "Dir":"北风",
                        "Power":"10-11级"
                    },
                    "9":{
                        "Dir":"旋转不定",
                        "Power":"11-12级"
                    }
                },
                "weatherCityIcon":{
                    "default":{
                        "big":"http://mat1.gtimg.com/weather/weatherIco/midImg/",
                        "mid":"http://mat1.gtimg.com/www/images/qq2012/weather/icon35/",
                        "small":"http://mat1.gtimg.com/www/images/qq2012/weather/"
                    },
                    "00":{
                        "icon":"m0.png",
                        "weatherTxt":"晴"
                    },
                    "01":{
                        "icon":"m1.png",
                        "weatherTxt":"多云"
                    },
                    "02":{
                        "icon":"m2.png",
                        "weatherTxt":"阴"
                    },
                    "03":{
                        "icon":"m3.png",
                        "weatherTxt":"阵雨"
                    },
                    "04":{
                        "icon":"m4.png",
                        "weatherTxt":"雷阵雨"
                    },
                    "05":{
                        "icon":"m4.png",
                        "weatherTxt":"雷阵雨并伴有冰雹"
                    },
                    "06":{
                        "icon":"m6.png",
                        "weatherTxt":"雨夹雪"
                    },
                    "07":{
                        "icon":"m7.png",
                        "weatherTxt":"小雨"
                    },
                    "08":{
                        "icon":"m8.png",
                        "weatherTxt":"中雨"
                    },
                    "09":{
                        "icon":"m9.png",
                        "weatherTxt":"大雨"
                    },
                    "10":{
                        "icon":"m10.png",
                        "weatherTxt":"暴雨"
                    },
                    "11":{
                        "icon":"m17.png",
                        "weatherTxt":"大暴雪"
                    },
                    "12":{
                        "icon":"m17.png",
                        "weatherTxt":"特大暴雪"
                    },
                    "13":{
                        "icon":"m14.png",
                        "weatherTxt":"阵雪"
                    },
                    "14":{
                        "icon":"m14.png",
                        "weatherTxt":"小雪"
                    },
                    "15":{
                        "icon":"m16.png",
                        "weatherTxt":"中雪"
                    },
                    "16":{
                        "icon":"m16.png",
                        "weatherTxt":"大雪"
                    },
                    "17":{
                        "icon":"m17.png",
                        "weatherTxt":"暴雪"
                    },
                    "18":{
                        "icon":"m18.png",
                        "weatherTxt":"雾"
                    },
                    "19":{
                        "icon":"m6.png",
                        "weatherTxt":"冻雨"
                    },
                    "20":{
                        "icon":"m20.png",
                        "weatherTxt":"沙尘暴"
                    },
                    "21":{
                        "icon":"m8.png",
                        "weatherTxt":"小雨-中雨"
                    },
                    "22":{
                        "icon":"m9.png",
                        "weatherTxt":"中雨-大雨"
                    },
                    "23":{
                        "icon":"m10.png",
                        "weatherTxt":"大雨-暴雨"
                    },
                    "24":{
                        "icon":"m10.png",
                        "weatherTxt":"暴雨-大暴雨"
                    },
                    "25":{
                        "icon":"m10.png",
                        "weatherTxt":"大暴雨-特大暴雨"
                    },
                    "26":{
                        "icon":"m16.png",
                        "weatherTxt":"小雪-中雪"
                    },
                    "27":{
                        "icon":"m16.png",
                        "weatherTxt":"中雪-大雪"
                    },
                    "28":{
                        "icon":"m17.png",
                        "weatherTxt":"大雪-暴雪"
                    },
                    "29":{
                        "icon":"m29.png",
                        "weatherTxt":"浮尘"
                    },
                    "30":{
                        "icon":"m29.png",
                        "weatherTxt":"扬沙"
                    },
                    "31":{
                        "icon":"m20.png",
                        "weatherTxt":"强沙尘暴"
                    },
                    "32":{
                        "icon":"m9.png",
                        "weatherTxt":"飑"
                    },
                    "33":{
                        "icon":"m33.png",
                        "weatherTxt":"龙卷风"
                    },
                    "34":{
                        "icon":"m14.png",
                        "weatherTxt":"弱高吹雪"
                    },
                    "35":{
                        "icon":"m18.png",
                        "weatherTxt":"轻雾"
                    },
                    "53":{
                        "icon":"m29.png",
                        "weatherTxt":"霾"
                    }
                },
                "living":{
                    'cl': [{ 'type': '', 'info': '' }, { 'type': '适宜', 'info': '天气晴朗，空气清新，是您晨练的大好时机。' }, { 'type': '较适宜', 'info': '较适宜晨练，某些气象条件会对晨练产生一定影响，但影响不大。' }, { 'type': '较不宜', 'info': '某些气象因素对晨练造成不利影响，较不宜晨练。' }, { 'type': '不适宜', 'info': '气象因素非常不利于室外锻炼，请尽量避免户外晨练。'}],
                    'cy': [{ 'type': '', 'info': '' }, { 'type': '炎热', 'info': '薄型T恤衫。' }, { 'type': '热舒适', 'info': '短套装、T恤夏季服装。' }, { 'type': '舒适', 'info': '长袖服装。' }, { 'type': '凉舒适', 'info': '薄型套装等春秋过渡装。'},{ 'type': '温凉', 'info': '夹衣或西服套装加薄羊毛衫。' }, { 'type': '凉', 'info': '厚外套加毛衣等春秋服装。' }, { 'type': '冷', 'info': '棉衣加羊毛衫等冬季服装。' }, { 'type':'寒冷','info': '厚羽绒服等隆冬服装。'}],
                    'gm': [{ 'type': '', 'info': '' }, { 'type': '少发', 'info': '各项气象条件适宜，发生感冒机率较低。' }, { 'type': '较易发', 'info': '较易发生感冒，体质较弱的朋友请注意适当防护。' }, { 'type': '易发', 'info': '发生感冒机率较大，请加强自我防护避免感冒。'}, { 'type': '极易发', 'info': '极易发生感冒，请特别注意增加衣服保暖防寒避免感冒。'}],
                    'xc': [{ 'type': '', 'info': '' }, { 'type': '适宜', 'info': '适宜洗车，未来持续两天无雨天气较好，适合擦洗汽车。' }, { 'type': '较适宜', 'info': '较适宜洗车，未来一天无雨，风力较小，较适合擦洗汽车。' }, { 'type': '较不宜', 'info': '较不宜洗车，擦洗一新的汽车可能会蒙上污垢。'}, { 'type': '不宜', 'info': '不宜洗车，路上的泥水可能会再次弄脏您的爱车。'}],
                    'zs': [{ 'type': '无', 'info': '温度不高，其他各项气象条件适宜，中暑机率极低。' }, { 'type': '少发', 'info': '气温偏高，有可能中暑，体质较弱的朋友请注意防暑降温，避免长时间在日光下暴晒或在高温环境中工作。' }, { 'type': '较易', 'info': '气温较高，较易中暑，体弱者请避免长时间在日光下暴晒或在高温环境中工作。' }, { 'type': '容易', 'info': '气温很高，热气逼人，容易中暑，请注意防暑降温，避免长时间在日光下暴晒或在高温环境中工作。'}, { 'type': '极易', 'info': '气温极高，热浪滚滚，极易中暑，请注意防暑降温，避免在高温环境中工作。'}],
                    'zwx': [{ 'type': '', 'info': '' }, { 'type': '最弱', 'info': '属弱紫外线辐射天气，无需特别防护。若长期在户外，建议涂擦SPF在8-12之间的防晒护肤品。' }, { 'type': '弱', 'info': '紫外线强度较弱，建议出门前涂擦SPF在12-15之间、PA+的防晒护肤品。' }, { 'type': '中等', 'info': '属中等强度紫外线辐射天气，外出时建议涂擦SPF高于15、PA+的防晒护肤品，戴帽子、太阳镜。'}, { 'type': '强', 'info': '紫外线辐射强，建议涂擦SPF20左右、PA++的防晒护肤品。避免在10点至14点暴露于日光下。'}, { 'type': '很强', 'info': '紫外线辐射极强，建议涂擦SPF20以上、PA++的防晒护肤品，尽量避免暴露于日光下。'}],
                    'ls': [{ 'type': '', 'info': '' }, { 'type': '极适宜', 'info': '极适宜晾晒，请抓紧时机晾晒。' }, { 'type': '适宜', 'info': '适宜晾晒，赶紧把久未见阳光的衣物搬出来吸收一下太阳的味道吧！' }, { 'type': '基本适宜', 'info': '午后温暖的阳光仍能满足你驱潮消霉杀菌的晾晒需求。'}, { 'type': '不太适宜', 'info': '气象要素对晾晒有影响，不太适宜晾晒。若非晾晒不可，请尽量选择通风的地点。'}, { 'type': '不适宜', 'info': '不适宜晾晒。如果非晾晒不可，请在室内进行并关牢门窗。'}],
                    'kqwr': [{ 'type': '', 'info': '' }, { 'type': '优', 'info': '非常有利于空气污染物稀释、扩散和清除，可在室外正常活动。' }, { 'type': '良', 'info': '有利于空气污染物稀释、扩散和清除，可在室外正常活动。' }, { 'type': '中', 'info': '对空气污染物稀释、扩散和清除无明显影响，易感人群应适当减少室外活动时间。'}, { 'type': '较差', 'info': '较不利于空气污染物稀释、扩散和清除，请适当减少室外活动时间。'}, { 'type': '差', 'info': '不利于空气污染物稀释、扩散和清除，请尽量避免在室外长时间活动。'}],
                    'ly': [{ 'type': '', 'info': '' }, { 'type': '适宜', 'info': '明媚阳和微风伴您一路同行，适宜旅游。' }, { 'type': '较适宜', 'info': '虽然会多云或有风，但仍是出行游玩的好时机，较适宜旅游。' }, { 'type': '一般', 'info': '可能出现阴天、大风、阵雨或沙尘等天气，将会给出行带来一些不便，旅游指数一般。'}, { 'type': '较不宜', 'info': '可能出现雨雪，强风，大雾等天气，人体在户外会感觉不舒适，将会给出行带来很多不便，较不适宜出游。'}, { 'type': '不适宜', 'info': '可能出现较大的雨雪，强风，冻雨，沙尘暴等天气，对出行造成一定困难，不适宜旅游。'}],
                    'kt': [{ 'type': '', 'info': '' },{ 'type': '长时间开启', 'info': '闷热，您需长时间开启制冷空调来降温除湿。' }, { 'type': '部分时间开启', 'info': '天气热，建议您在适当的时候开启制冷空调来降低温度。' }, { 'type': '少部分时间', 'info': '中午的时候您将会感到有点热，因此建议在午后较热时开启制冷空调。' }, { 'type': '较少开启', 'info': '您将感到很舒适，一般不需要开启空调。'}, { 'type': '开启制暖空调', 'info': '冷,适当开启制暖空调调节室内温度。'}],
                    'dy': [{ 'type': '', 'info': '' }, { 'type': '适宜', 'info': '非常适宜垂钓，风和日丽的天气将陪伴你度过愉快的垂钓时光。' }, { 'type': '较适宜', 'info': '较适合垂钓，气象条件对垂钓会产生一定影响，但影响不大。' }, { 'type': '不适宜', 'info': '天气不好，不适合垂钓，请选择别的娱乐方式。'}],
                    'fs': [{ 'type': '', 'info': '' },{ 'type': '弱', 'info': '长期在户外，建议涂擦SPF在8-12之间的防晒护肤品。' }, { 'type': '较弱', 'info': '建议涂擦SPF在12-15之间，PA+的防晒护肤品。' }, { 'type': '中等', 'info': '建议涂擦SPF指数高于15,PA+的防晒护肤品。' }, { 'type': '强', 'info': '外出时应加强防护，建议涂擦SPF在15—20之间PA++的防晒护肤品。'}, { 'type': '极强', 'info': '外出时应特别加强防护，建议涂擦SPF20以上，PA++的防晒护肤品，并随时补涂。'}],
                    'gj': [{ 'type': '适宜', 'info': '天气条件很适合逛街，可以尽情享受逛街的乐趣。' }, { 'type': '较适宜', 'info': '气象要素对逛街有一定影响，比较适宜逛街。' }, { 'type': '较不宜', 'info': '天气条件不适宜逛街，应尽量避免外出逛街。' }, { 'type': '不适宜', 'info': '气象条件不适合逛街，请尽量选择室内活动。'}],
                    'hc': [{ 'type': '', 'info': '' }, { 'type': '适宜', 'info': '天气晴朗，温度适宜，非常适合划船或嬉玩各种水上运动。' }, { 'type': '较适宜', 'info': '较适宜划船，有些气象因素对划船会产生些影响，但影响不大。' }, { 'type': '不适宜', 'info': '不适宜划船，建议选择别的娱乐方式或采取必要措施。'}],
                    'jt': [{ 'type': '', 'info': '' },{ 'type': '良好', 'info': '交通气象条件良好，车辆可以正常行驶。' }, { 'type': '较好', 'info': '交通气象条件较好，但不适宜高速行驶，司机应更加集中注意力，保持车距。' }, { 'type': '一般', 'info': '交通气象条件一般，刹车距离延长，事故易发期，注意车距，务必小心驾驶。' }, { 'type': '较差', 'info': '交通气象条件较差，事故高发期，车辆应低速行驶。'}, { 'type': '很差', 'info': '交通气象条件很差，车辆行驶缓慢，尽量减少出行。'}],
                    'ys': [{ 'type': '', 'info': '' }, { 'type': '不带伞', 'info': '降水概率很低，因此您在出门的时候无须带雨伞。' }, { 'type': '带伞', 'info': '将有阵雨或阵雪，如果您要短时间外出的话可不必带雨伞。' }, { 'type': '带伞', 'info': '有小雨或小雪或中雪，在短时间外出可收起雨伞，但最好还是带上雨伞。'}, { 'type': '带伞', 'info': '会有较大的雨雪天气，您在外出的时候一定要带雨伞，以免被雨水淋湿。'}],
                    'mf': [{ 'type': '', 'info': '' }, { 'type': '极适宜', 'info': '各项气象条件都适宜美发，这为您的头发创造一个健康、洁净的生长环境。' }, { 'type': '适宜', 'info': '有某项气象因素会影响您的秀发生长，请注意保养。 ' }, { 'type': '一般', 'info': '气象条件对美发有一定影响，注意保养您的美发，细心呵护！'}],
                    'ysh': [{ 'type': '', 'info': '' }, { 'type': '适宜', 'info': '天气晴朗，你可以尽情外出享受夜生活的乐趣，不用担心天气会来捣乱。' }, { 'type': '较适宜', 'info': '虽然有风或有雨雪天气出现，但只要提前有所准备，您仍然可以享受夜生活的乐趣。' }, { 'type': '较不宜', 'info': '天气会使人体在户外感觉不舒适，建议夜生活最好在室内进行。'}],
                    'ffz': [{ 'type': '', 'info': '' }, { 'type': '适宜', 'info': '这种天气去放风筝既可以舒展筋骨，又可放松身心。' }, { 'type': '较适宜', 'info': '选择合适的地点，还是较适宜放风筝的。' }, { 'type': '不宜', 'info': '气象条件不适宜放风筝。'}],
                    'hz': [{ 'type': '', 'info': '' }, { 'type': '保湿防龟裂', 'info': '天气寒冷，多补水，选用滋润保湿型化妆品，使用润唇膏。' }, { 'type': '保湿', 'info': '皮肤易缺水，用保湿型霜类化妆品，使用润唇膏。' }, { 'type': '控油', 'info': '建议用露质面霜打底，水质无油粉底霜，透明粉饼，粉质胭脂。'}, { 'type': '防晒', 'info': '天气炎热，易出汗，建议使用防脱水防晒指数高的化妆品，经常补粉。'}],
                    'fh': [{ 'type': '', 'info': '' }, { 'type': '凉', 'info': '室外活动注意适当增减衣物。' }, { 'type': '冷', 'info': '室外活动要穿厚实一点，年老体弱者要适当注意保暖。' }, { 'type': '寒冷', 'info': '室外活动要注意保暖，可戴手套与帽子。' }, { 'type': '非常寒冷', 'info': '室外活动注意保暖防寒，可戴厚手套和帽子，年老体弱者避免长时间外出。' }, { 'type': '严寒', 'info': '着羽绒服、皮大衣仍感到寒冷，室外活动须戴厚棉、皮手套和帽子。' }, { 'type': '冰冻严寒', 'info': '极易造成裸露皮肤冻伤，尽量避免野外作业和外出。' }, { 'type': '', 'info': '' }, { 'type': '微凉', 'info': '温度未达到风寒所需的低温，稍作防寒准备即可。'}],
                    'xq': [{ 'type': '', 'info': '' }, { 'type': '好', 'info': '天气晴朗，阳光灿烂，空气温润，和风飘飘，美好的天气会带来一天接踵而来的好心情。' }, { 'type': '较好', 'info': '温度舒适，您会觉得精神振奋，意气风发，心情舒畅。' }, { 'type': '较差', 'info': '天气阴沉或有雾，会感觉莫名的压抑，情绪低落。'}, { 'type': '差', 'info': '天气阴沉有雨或有沙尘，闷热潮湿的空气会让人感到胸闷，心情糟糕。'}],
                    'yd': [{ 'type': '', 'info': '' }, { 'type': '适宜', 'info': '天气较好，且紫外线辐射不强，适宜户外运动。' }, { 'type': '较适宜', 'info': '较适宜在户内低强度运动，户外运动需防晒避风。' }, { 'type': '较不宜', 'info': '受大风，气压，强紫外线，过高或过低的气温等天气影响，较不宜运动'}],
                    'yh': [{ 'type': '', 'info': '' }, { 'type': '适宜', 'info': '天气晴朗，风和日丽，适宜与情人约会。' }, { 'type': '较适宜', 'info': '天气不会有太大的影响，你仍然可以有一个愉快的约会。' }, { 'type': '较不适宜', 'info': '会有影响人体舒适的不好天气出现，室外约会可能会让恋人们受些苦，最好在温暖的室内促膝谈心'}, { 'type': '不适宜', 'info': '天气会对人体有较大影响，外出约会还可能会败兴而归，男士请别约美眉逛街。'}]
                }
            }

            if(window.location.toString().indexOf('type=weather') != -1){
            }else{
                try{
                    QQ.LoadScript("http://weather.gtimg.cn/city/" + weatherId + ".js?ref=qqIndex&ran="+Math.random(),function(){
                        var weatherCityName = __weather_city["bi_name"];
                        if(weatherCityName == "西安(陕西)"){
                            weatherCityName = "西安";
                        }else if(weatherCityName == "邢台市"){
                            weatherCityName = "邢台";
                        }else if(weatherCityName == "绍兴市"){
                            weatherCityName = "绍兴";
                        }else if(weatherCityName == "襄樊"){
                            weatherCityName = "襄阳";
                        }
                        var weatherTemperatureMath = __weather_city["sk_tp"];
                        var weatherCityIcon = weatherCityMap["weatherCityIcon"]["default"]["big"]+weatherCityMap["weatherCityIcon"][__weather_city["sk_wt"]]["icon"];
                        var weatherCityTxt = weatherCityMap["weatherCityIcon"][__weather_city["sk_wt"]]["weatherTxt"];
                        var weatherCityWindDir = weatherCityMap["weatherCityWind"][__weather_city["sk_wd"]]["Dir"];
                        var weatherCityWindPower = weatherCityMap["weatherCityWind"][__weather_city["sk_wp"]]["Power"];
                      	nowTimeHour = nowTimeHour + 8;
                        if(nowTimeHour<20 && nowTimeHour >4){
                            var weatherIcon = weatherMap["weatherIcon"][__weather_city.sk_wt]["day"];
                        }else{
                            var weatherIcon = weatherMap["weatherIcon"][__weather_city.sk_wt]["night"];
                        }
                        //if(__weather_city.wk[0][0].ts.substring(8,10) == __weather_city.wk[0][0].te.substring(8,10)){
                        //alert("ts="+new Date(("2013-5-2 8:00:00").replace(/\-/g, "/")).getTime());
                        //alert("te="+parseInt(new Date(__weather_city.wk[0][0].te).getTime()/1000/3600/24));
                        if(parseInt(new Date(__weather_city.wk[0][0].ts.replace(/\-/g, "/")).getTime()/1000/3600/24) == parseInt(new Date(__weather_city.wk[0][0].te.replace(/\-/g, "/")).getTime()/1000/3600/24)){
                            if(__weather_city.wk[0][0].tmin!="NULL"){
                                var weatherFutureTemperature01 = __weather_city.wk[0][0].tmin;
                            }else{
                                var weatherFutureTemperature01 = __weather_city.wk[0][0].tmax;
                            }
                            var weatherFutureIcon01 = weatherCityMap["weatherCityIcon"]["default"]["small"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][0].wt]["icon"];
                            var weatherMoreIcon01 = weatherCityMap["weatherCityIcon"]["default"]["mid"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][0].wt]["icon"];
                            var weatherFutureTxt01 = weatherCityMap["weatherCityIcon"][__weather_city.wk[0][0].wt]["weatherTxt"];

                            if(__weather_city.wk[0][1].tmin!="NULL"){
                                var weatherFutureTemperature02 = __weather_city.wk[0][1].tmin;
                            }else{
                                var weatherFutureTemperature02 = __weather_city.wk[0][1].tmax;
                            }
                            var weatherFutureIcon02 = weatherCityMap["weatherCityIcon"]["default"]["small"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][1].wt]["icon"];
                            var weatherFutureTxt02 = weatherCityMap["weatherCityIcon"][__weather_city.wk[0][1].wt]["weatherTxt"];

                            if(__weather_city.wk[0][2].tmin!="NULL"){
                                var weatherFutureTemperature03 = __weather_city.wk[0][2].tmin;
                            }else{
                                var weatherFutureTemperature03 = __weather_city.wk[0][2].tmax;
                            }
                            var weatherFutureIcon03 = weatherCityMap["weatherCityIcon"]["default"]["small"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][2].wt]["icon"];
                            var weatherMoreIcon03 = weatherCityMap["weatherCityIcon"]["default"]["mid"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][2].wt]["icon"];
                            var weatherFutureTxt03 = weatherCityMap["weatherCityIcon"][__weather_city.wk[0][2].wt]["weatherTxt"];

                            if(__weather_city.wk[0][3].tmin!="NULL"){
                                var weatherFutureTemperature04 = __weather_city.wk[0][3].tmin;
                            }else{
                                var weatherFutureTemperature04 = __weather_city.wk[0][3].tmax;
                            }
                            var weatherFutureIcon04 = weatherCityMap["weatherCityIcon"]["default"]["small"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][3].wt]["icon"];
                            var weatherFutureTxt04 = weatherCityMap["weatherCityIcon"][__weather_city.wk[0][3].wt]["weatherTxt"];

                            if(__weather_city.wk[0][4].tmin!="NULL"){
                                var weatherFutureTemperature05 = __weather_city.wk[0][4].tmin;
                            }else{
                                var weatherFutureTemperature05 = __weather_city.wk[0][4].tmax;
                            }
                            var weatherFutureIcon05 = weatherCityMap["weatherCityIcon"]["default"]["small"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][4].wt]["icon"];
                            var weatherMoreIcon05 = weatherCityMap["weatherCityIcon"]["default"]["mid"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][3].wt]["icon"];
                            var weatherFutureTxt05 = weatherCityMap["weatherCityIcon"][__weather_city.wk[0][4].wt]["weatherTxt"];

                            if(__weather_city.wk[0][5].tmin!="NULL"){
                                var weatherFutureTemperature06 = __weather_city.wk[0][5].tmin;
                            }else{
                                var weatherFutureTemperature06 = __weather_city.wk[0][5].tmax;
                            }
                            var weatherFutureIcon06 = weatherCityMap["weatherCityIcon"]["default"]["small"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][5].wt]["icon"];
                            var weatherFutureTxt06 = weatherCityMap["weatherCityIcon"][__weather_city.wk[0][5].wt]["weatherTxt"];
                        }else{
                            var weatherFutureTemperature01 = __weather_city.sk_tp;
                            var weatherMoreIcon01 = weatherCityMap["weatherCityIcon"]["default"]["mid"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][0].wt]["icon"];
                            var weatherFutureTxt01 = weatherCityMap["weatherCityIcon"][__weather_city.sk_wt]["weatherTxt"];
                            if(__weather_city.wk[0][0].tmin!="NULL"){
                                var weatherFutureTemperature02 = __weather_city.wk[0][0].tmin;
                            }else{
                                var weatherFutureTemperature02 = __weather_city.wk[0][0].tmax;
                            }
                            var weatherFutureIcon02 = weatherCityMap["weatherCityIcon"]["default"]["small"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][0].wt]["icon"];
                            var weatherFutureTxt02 = weatherCityMap["weatherCityIcon"][__weather_city.wk[0][0].wt]["weatherTxt"];

                            if(__weather_city.wk[0][1].tmin!="NULL"){
                                var weatherFutureTemperature03 = __weather_city.wk[0][1].tmin;
                            }else{
                                var weatherFutureTemperature03 = __weather_city.wk[0][1].tmax;
                            }
                            var weatherFutureIcon03 = weatherCityMap["weatherCityIcon"]["default"]["small"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][1].wt]["icon"];
                            var weatherMoreIcon03 = weatherCityMap["weatherCityIcon"]["default"]["mid"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][2].wt]["icon"];
                            var weatherFutureTxt03 = weatherCityMap["weatherCityIcon"][__weather_city.wk[0][1].wt]["weatherTxt"];

                            if(__weather_city.wk[0][2].tmin!="NULL"){
                                var weatherFutureTemperature04 = __weather_city.wk[0][2].tmin;
                            }else{
                                var weatherFutureTemperature04 = __weather_city.wk[0][2].tmax;
                            }
                            var weatherFutureIcon04 = weatherCityMap["weatherCityIcon"]["default"]["small"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][2].wt]["icon"];
                            var weatherFutureTxt04 = weatherCityMap["weatherCityIcon"][__weather_city.wk[0][2].wt]["weatherTxt"];

                            if(__weather_city.wk[0][3].tmin!="NULL"){
                                var weatherFutureTemperature05 = __weather_city.wk[0][3].tmin;
                            }else{
                                var weatherFutureTemperature05 = __weather_city.wk[0][3].tmax;
                            }
                            var weatherFutureIcon05 = weatherCityMap["weatherCityIcon"]["default"]["small"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][3].wt]["icon"];
                            var weatherMoreIcon05 = weatherCityMap["weatherCityIcon"]["default"]["mid"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][3].wt]["icon"];
                            var weatherFutureTxt05 = weatherCityMap["weatherCityIcon"][__weather_city.wk[0][3].wt]["weatherTxt"];

                            if(__weather_city.wk[0][4].tmin!="NULL"){
                                var weatherFutureTemperature06 = __weather_city.wk[0][4].tmin;
                            }else{
                                var weatherFutureTemperature06 = __weather_city.wk[0][4].tmax;
                            }
                            var weatherFutureIcon06 = weatherCityMap["weatherCityIcon"]["default"]["small"]+weatherCityMap["weatherCityIcon"][__weather_city.wk[0][4].wt]["icon"];
                            var weatherFutureTxt06 = weatherCityMap["weatherCityIcon"][__weather_city.wk[0][4].wt]["weatherTxt"];
                        }
                        $("#ipWeather").html(weatherCityName);
                        $("#ipWeather").width($("#ipWeather").html().length*12);
                        $("#weatherTemperature").html(weatherTemperatureMath + "℃");
                        $("#weatherIcon").attr("style","background:url("+weatherIcon+") no-repeat;_background:none;_filter:progid:DXImageTransform.Microsoft.AlphaImageLoader(enabled=true,sizingMethod=scale,src='"+weatherIcon+"');");
                        $("#weatherLayout").width($("#weatherTemperature").width()+$("#weatherIcon").width()+$("#ipWeather").width()+20);
                        $("#weatherLayout").show();

                        $("#weatherMoreCity").html(weatherCityName);
                        $("#weatherMoreTxt").html(weatherCityTxt);
                        if(__weather_city.zs_kqwr > 5){
                            __weather_city.zs_kqwr = 5;
                        }
                        $("#weatherMoreAirTxt").html(weatherCityMap["living"]["kqwr"][__weather_city.zs_kqwr]["type"]);
                        $("#weatherMoreTem").html(weatherTemperatureMath + "℃");

                        $("#weatherMoreTodayIcon").attr("style","background:url("+weatherMoreIcon01+") no-repeat;_background:none;_filter:progid:DXImageTransform.Microsoft.AlphaImageLoader(enabled=true,sizingMethod=scale,src='"+weatherMoreIcon01+"');");
                        $("#weatherMoreTodayHighest").html(weatherFutureTemperature01);
                        $("#weatherMoreTodayLowest").html(weatherFutureTemperature02);
                        $("#weatherMoreTomorrowIcon").attr("style","background:url("+weatherMoreIcon03+") no-repeat;_background:none;_filter:progid:DXImageTransform.Microsoft.AlphaImageLoader(enabled=true,sizingMethod=scale,src='"+weatherMoreIcon03+"');");
                        $("#weatherMoreTomorrowHighest").html(weatherFutureTemperature03);
                        $("#weatherMoreTomorrowLowest").html(weatherFutureTemperature04);
                        $("#weatherMoreAfterTomorrowIcon").attr("style","background:url("+weatherMoreIcon05+") no-repeat;_background:none;_filter:progid:DXImageTransform.Microsoft.AlphaImageLoader(enabled=true,sizingMethod=scale,src='"+weatherMoreIcon05+"');");
                        $("#weatherMoreAfterTomorrowHighest").html(weatherFutureTemperature05);
                        $("#weatherMoreAfterTomorrowLowest").html(weatherFutureTemperature06);

                        if(type == 0){
                            weatherCityNameDefault = weatherCityName;
                            weatherTemperatureMathDefault = weatherTemperatureMath;
                            weatherIconDefault = weatherIcon;
                            weatherCityTxtDefault = weatherCityTxt;
                            weatherTemperatureMathDefault = weatherTemperatureMath;
                            weatherMoreIcon01Default = weatherMoreIcon01;
                            weatherFutureTemperature01Default = weatherFutureTemperature01;
                            weatherFutureTemperature02Default = weatherFutureTemperature02;
                            weatherMoreIcon03Default = weatherMoreIcon03;
                            weatherFutureTemperature03Default = weatherFutureTemperature03;
                            weatherFutureTemperature04Default = weatherFutureTemperature04;
                            weatherMoreIcon05Default = weatherMoreIcon05;
                            weatherFutureTemperature05Default = weatherFutureTemperature05;
                            weatherFutureTemperature06Default = weatherFutureTemperature06;
                            weatherMoreAirTxtDefault = $("#weatherMoreAirTxt").html();
                        }
                    });
                }catch(e){}
            }
            //http://weather.gtimg.cn/qqindex/01010101.js?ref=qqIndex
        });
    },
    locationCustomInit:function(proName,cityName){
        var weatherMoreCitySelectUlArr = [];
        $("#ipSetProvince").html(proName);
        $("#weatherMoreCitySelectUl").html("");
        for(var key in cityCode[proName]){
            weatherMoreCitySelectUlArr.push("<li><a href='javascript:void(0);'>"+key+"</a></li>");
        }
        $("#weatherMoreCitySelectUl").html(weatherMoreCitySelectUlArr.join(""));
        $("#ipSetCity").html($("#weatherMoreCitySelectUl").children().first().children().first().html());
    },
    locationCustom:function(proName,cityName,defaultPro,defaultCity){
        var weatherMoreOverTime = "";
        var weatherMoreOutTime = "";
        var weatherMoreProviceSelectBind = "";
        var weatherMoreCitySelectBind = "";

        $("#weatherLayout").bind("mouseover",function(){
            weatherMoreOverTime = setTimeout(function(){
                if($("#weatherMore").is(":hidden") == true){

                }
                $("#weatherMore").show();
            },500);
            if(weatherMoreOutTime){
                clearTimeout(weatherMoreOutTime);
            }
        });
        $("#weatherLayout").bind("mouseout",function(){
            if(weatherMoreOverTime){
                clearTimeout(weatherMoreOverTime);
            }
            weatherMoreOutTime = setTimeout(function(){
                $("#weatherMoreProviceSelect").hide();
                $("#weatherMoreCitySelect").hide();
                $("#weatherMore").hide();
                $("#weatherMore").removeClass("weatherMoreFlipped");
            },1000);
        });
        $("#weatherMoreReset").bind("click",function(){
            $("#weatherMore").removeClass("weatherMoreFlipped");
            ipMe.weathercheck(defaultPro,defaultCity,1);
            ipMe.locationCustomInit(defaultPro,defaultCity);
            ipMe.newscheck(defaultPro);
            ipMe.housecheck(defaultPro,defaultCity);
            //ipMe.autocheck(defaultPro);
            ipMe.autoPriceCheck(defaultCity);
            ipMe.tuancheck(defaultPro,defaultCity);
            ipMe.meishicheck(defaultPro);

            QQ.localData.remove("customProvince");
            QQ.localData.remove("customCity");
            QQ.localData.remove("customNewsProvince");
            QQ.localData.remove("customNewsCity");

            updateStatus();
        });
        $("#ipSetProvince").bind("click",function(e){
            e = e||window.event;
            var target = e.target||e.srcElement;
            if('function' === typeof e.preventDefault){
                e.preventDefault();
                e.stopPropagation();
            }else{
                e.returnValue = false;
                e.cancelBubble = true;
            }
            $("#weatherMoreProviceSelect").show();
            setTimeout(function(){
                weatherMoreProviceSelectBind = $(document).bind("_click",function(){
                    $("#weatherMoreProviceSelect").hide();
                    if(weatherMoreProviceSelectBind){
                        $(document).unbind("click",weatherMoreProviceSelectBind);
                    }
                });
            },100);
        });

        $("#ipSetCity").bind("click",function(e){
            e = e||window.event;
            var target = e.target||e.srcElement;
            if('function' === typeof e.preventDefault){
                e.preventDefault();
                e.stopPropagation();
            }else{
                e.returnValue = false;
                e.cancelBubble = true;
            }
            $("#weatherMoreCitySelect").show();
            setTimeout(function(){
                weatherMoreCitySelectBind = $(document).bind("_click",function(){
                    $("#weatherMoreCitySelect").hide();
                    if(weatherMoreCitySelectBind){
                        $(document).unbind("click",weatherMoreCitySelectBind);
                    }
                });
            },100);
        });

        $("#weatherMoreProviceSelect").bind("click",function(e){
            e = e||window.event;
            var target = e.target||e.srcElement;
            if('A' !== target.nodeName){return false;}
            $("#weatherMoreProviceSelect").hide();
            $("#ipSetProvince").html(target.innerHTML);
            $("#weatherMoreCitySelectUl").html("");
            for(var key in cityCode[target.innerHTML]){
                $("#weatherMoreCitySelectUl").append("<li><a href='javascript:void(0);'>"+key+"</a></li>");
            }
            $("#ipSetCity").html($("#weatherMoreCitySelectUl").children().first().children().first().html());
        });

        $("#weatherMoreCitySelectUl").bind("click",function(e){
            e = e||window.event;
            var target = e.target||e.srcElement;
            if('A' !== target.nodeName){return false;}
            $("#weatherMoreCitySelect").hide();
            $("#ipSetCity").html(target.innerHTML);
        });

        $("#weatherMoreSet").bind("click",function(){
            $("#weatherMore").addClass("weatherMoreFlipped");
        });

        $("#weatherMoreNewsCheckbox").bind("click",function(){
            if($("#weatherMoreNewsCheckbox").hasClass("weatherMoreNewsYes")){
                $("#weatherMoreNewsCheckbox").removeClass("weatherMoreNewsYes");
            }else{
                $("#weatherMoreNewsCheckbox").addClass("weatherMoreNewsYes");
            }
        });

        $("#weatherMoreSubmit").bind("click",function(){
            $("#weatherMore").removeClass("weatherMoreFlipped");
            var personalProvince = $("#ipSetProvince").html();
            ipProvince = $("#ipSetProvince").html();
            var personalCity = $("#ipSetCity").html();
            if($("#weatherMoreNewsCheckbox").hasClass("weatherMoreNewsYes")){
                ipMe.weathercheck(personalProvince,personalCity);
                ipMe.newscheck(personalProvince);
                ipMe.housecheck(personalProvince,personalCity);
                //ipMe.autocheck(personalProvince);
           		ipMe.autoPriceCheck(defaultCity);
                ipMe.tuancheck(personalProvince,personalCity);
                ipMe.meishicheck(personalProvince);
                QQ.localData.set("customNewsProvince", personalProvince);
                QQ.localData.set("customNewsCity", personalCity);
            }else{
                ipMe.weathercheck(personalProvince,personalCity);
            }
            QQ.localData.set("customProvince", personalProvince);
            QQ.localData.set("customCity", personalCity);

            updateStatus();
        });
        $("#weatherMoreCancel").bind("click",function(){
            $("#weatherMore").removeClass("weatherMoreFlipped");
        });
    },
    tuancheck:function(proName,cityName){
        /*if(cityName == "天津市"){ //天津生活

         var ajaxTianjinLive = new Ajax({
         headers:{"If-Modified-Since":0},
         url:"http://www.qq.com/site/tenpay/liveTianjin.htm",
         parameters:{"random":Math.random()},
         method:"get"
         });
         ajaxTianjinLive.onComplete = function(responseObject){
         $("#liveInfo").html(responseObject.textString);
         };
         ajaxTianjinLive.start();
         $("#tuangouH2").html("<a target='_blank' href='http://e.t.qq.com/TJB1009' bossZone='tjlifeTAB'>天津生活</a>");

         }else{
         */
        if(liveMap[proName]){ //地方站
            var ajaxTuan = new Ajax({
                headers:{"If-Modified-Since":0},
                url:liveMap[proName],
                parameters:{"random":Math.random()},
                method:"get"
            });
            ajaxTuan.onComplete = function(responseObject){
                $("#liveInfo").html(responseObject.textString);
            };
            ajaxTuan.start();
            $("#tuangouH2").html(tuanTitleMap[proName]);
        }else{ //非地方站
            //团购信息
            if(tuanMap[cityName]){
                tuanCity = encodeURI(tuanMap[cityName]).toLowerCase();
            }else{
                if(provincialCapitalMap[proName]){
                    tuanCity = encodeURI(provincialCapitalMap[proName]).toLowerCase();
                }else{
                    tuanCity = encodeURI("其他市").toLowerCase();
                }
            }
            var ajaxTuan = new Ajax({
                headers:{"If-Modified-Since":0},
                url:"http://www.qq.com/c/tuangouInfo20120917.htm",
                parameters:{"random":Math.random(),"city":tuanCity},
                method:"get"
            });
            ajaxTuan.onComplete = function(responseObject){
                $("#tuangouInfo").html(responseObject.textString);
            };
            ajaxTuan.start();

            //生活服务文字链
            var ajaxTuangouTxt = new Ajax({
                headers:{"If-Modified-Since":0},
                url:"/ninja/liveserverContent.htm",
                parameters:{"random":Math.random()},
                method:"get"
            });
            ajaxTuangouTxt.onComplete = function(responseObject){
                $("#tuangouTxt").html(responseObject.textString);
            };
            ajaxTuangouTxt.start();

            //生活服务标题
            $("#tuangouH2").html('<a href="http://vip.qq.com/life.html" target="_blank">生活服务</a>');
        }
        /*}*/
    },
    meishicheck:function(proName){
        if(meishiMap[proName]){
            var ajaxMeishiInfo = new Ajax({
                headers:{"If-Modified-Since":0},
                url:meishiMap[proName],
                parameters:{"random":Math.random()},
                method:"get"
            });
            ajaxMeishiInfo.onComplete = function(responseObject){
                $("#qqMeishi").html(responseObject.textString);
                $("#gongyi").hide();
                $("#qqMeishi").show();
            };
            ajaxMeishiInfo.start();
        }else{
            $("#qqMeishi").hide();
            $("#gongyi").show();
        }
    },
    housecheck:function(proName,cityName){
        houseUrl = "";
        if(houseMap[cityName]){
            houseUrl = houseMap[cityName];
        }else if(houseMap[proName]){
            houseUrl = houseMap[proName];
        }else{
            houseUrl = "/ninja/houseQuanguoContent_public.htm";
        }
        if(houseUrl != ""){
            var ajaxHouse = new Ajax({
                headers:{"If-Modified-Since":0},
                url:houseUrl,
                parameters:{"random":Math.random()},
                method:"get"
            });
            ajaxHouse.onComplete = function(responseObject){
                $("#house .bd").html(responseObject.textString);
                if($("#house").hasClass("contentModBig")){
                    $("#house .bd .contentRight").show();
                }else{
                    $("#house .bd .contentRight").hide();
                }
                if(houseTitleMap[cityName]){
                    $("#house .hd h2").html(houseTitleMap[cityName]);
                }else if(houseTitleMap[proName]){
                    $("#house .hd h2").html(houseTitleMap[proName]);
                }else{
                    $("#house .hd h2").html('<a href="http://house.qq.com" target="_blank" bossZone="housetab">房产</a>');
                }
                setTimeout(function(){
                    try{
                        $('#smartbox_form_1').empty();
                        createSmartbox("smartbox_form_1");
                    }catch(e){
                        QQ.LoadScript("http://mat1.gtimg.com/www/js/qq2012/smartboxRewrite1.7.js",function(){
                            $('#smartbox_form_1').empty();
                            createSmartbox("smartbox_form_1");
                        });
                    }
                },500);
            };
            ajaxHouse.start();
        }
    },
    autocheck:function(proName,cityName){
        autoUrl = "";
        if(autoMap[proName]){
            autoUrl = autoMap[proName];
        }
        if(autoUrl != ""){
            var ajaxAuto = new Ajax({
                headers:{"If-Modified-Since":0},
                url:autoUrl,
                parameters:{"random":Math.random()},
                method:"get"
            });
            ajaxAuto.onComplete = function(responseObject){
                $("#autonews").children().eq(5).html(responseObject.textString);
            };
            ajaxAuto.start();
        }
    },
    autoPriceCheck:function(cityName){
      	$("#autoLowPriceUl .name a").each(function(){
			var content = $(this).html().replace(/&quot;/gi,"");
			content = unescape(content.replace(/\\(u[0-9a-zA-Z]{4})/g, "%$1"));
			$(this).html(content)
		});
        if(autoPriceMap[cityName]){
	        autoPriceCity = encodeURI(autoPriceMap[cityName]).toLowerCase();
            var ajaxAutoPrice = new Ajax({
                headers:{"If-Modified-Since":0},
                url:"http://www.qq.com/c/autoLowPrice.htm",
                parameters:{"random":Math.random(),"city":autoPriceCity},
                method:"get"
            });
            ajaxAutoPrice.onComplete = function(responseObject){
              //$("#autoLowPriceUl").html(responseObject.textString.replace(/&quot;/gi,""));
              var content = responseObject.textString.replace(/&quot;/gi,"");
			  content = unescape(content.replace(/\\(u[0-9a-zA-Z]{4})/g, "%$1"));
              document.getElementById('autoLowPriceUl').innerHTML = content;
            };
            ajaxAutoPrice.start();
        }else{
            var ajaxAutoPrice = new Ajax({
                headers:{"If-Modified-Since":0},
                url:"http://www.qq.com/c/autoLowPrice.htm?city=%e5%85%a8%e5%9b%bd",
                parameters:{"random":Math.random()},
                method:"get"
            });
            ajaxAutoPrice.onComplete = function(responseObject){
              	//$("#autoLowPriceUl").html(responseObject.textString.replace("&quot;",""));
              	var content = responseObject.textString.replace(/&quot;/gi,"");
			  	content = unescape(content.replace(/\\(u[0-9a-zA-Z]{4})/g, "%$1"));
              	document.getElementById('autoLowPriceUl').innerHTML = content;
            };
            ajaxAutoPrice.start();
        }
    },
    init:function(){
        //拉取用户订制
        var param = encodeURI('{"busi_id":"tubd_sub_23","busi_subid":""}');
        QQ.LoadScript('http://i.match.qq.com/interface/getsub?callback=customOrder&para=' + param + '&random='+Math.random(),function(){});
        var _self = this;
        _self.pro = "北京市";
        _self.city = "北京市";
        var customProvince = QQ.localData.get("customProvince");
        var customCity = QQ.localData.get("customCity");
        var customNewsProvince = QQ.localData.get("customNewsProvince");
        var customNewsCity = QQ.localData.get("customNewsCity");
        customIpProvince = customNewsProvince;
        //var ipCookie = QQ.Cookie.get("ipCookie");

        $("#ipWeather").bind("click",function(){
            registerZone2({bossZone:'weather',url:encodeURI("http://weather.news.qq.com/qresult.htm?acity" + $("#ipWeather").html())},1);
            window.open(encodeURI("http://weather.news.qq.com/qresult.htm?acity" + $("#ipWeather").html()));
        });
        $("#weatherIcon").bind("click",function(){
            registerZone2({bossZone:'weather',url:encodeURI("http://weather.news.qq.com/qresult.htm?acity" + $("#ipWeather").html())},1);
            window.open(encodeURI("http://weather.news.qq.com/qresult.htm?acity" + $("#ipWeather").html()));
        });
        $("#weatherTemperature").bind("click",function(){
            registerZone2({bossZone:'weather',url:encodeURI("http://weather.news.qq.com/qresult.htm?acity" + $("#ipWeather").html())},1);
            window.open(encodeURI("http://weather.news.qq.com/qresult.htm?acity" + $("#ipWeather").html()));
        });
        $("#weatherMoreFutureCon").bind("click",function(){
            registerZone2({bossZone:'weather',url:encodeURI("http://weather.news.qq.com/qresult.htm?acity" + $("#ipWeather").html())},1);
            window.open(encodeURI("http://weather.news.qq.com/qresult.htm?acity" + $("#ipWeather").html()));
        });
        if(window.location.toString().indexOf('qq.com') != -1){
            QQ.LoadScript("http://fw.qq.com/ipaddress?ran="+Math.random(),function(){
                if (typeof(IPData) != "undefined") {
                    _self.ip = IPData[0];
                    _self.pro = IPData[2];
                    ipProvince = IPData[2];
                    switch(IPData[2]){
                        case '北京市':
                            _self.city = IPData[2];
                            break;
                        case '上海市':
                            _self.city = IPData[2];
                            break;
                        case '重庆市':
                            _self.city = IPData[2];
                            break;
                        case '天津市':
                            _self.city = IPData[2];
                            break;
                        case '香港':
                            _self.city = IPData[2];
                            break;
                        case '澳门':
                            _self.city = IPData[2];
                            break;
                        default:
                            _self.city = IPData[3];
                    }
                }
                _self.locationCustom(_self.pro,_self.city,_self.pro,_self.city);
                if(customProvince != null && customCity != null){
                    _self.weathercheck(customProvince,customCity);
                    _self.locationCustomInit(customProvince,customCity);
                }else{
                    _self.weathercheck(_self.pro,_self.city,0);
                    _self.locationCustomInit(_self.pro,_self.city);
                }
                if(customNewsProvince != null && customNewsCity != null){
                    ipProvince = customNewsProvince;
                    _self.newscheck(customNewsProvince);
                    _self.housecheck(customNewsProvince,customNewsCity);
                    //_self.autocheck(customNewsProvince);
            		_self.autoPriceCheck(customNewsCity);
                    _self.tuancheck(customNewsProvince,customNewsCity);
                    _self.meishicheck(customNewsProvince);
                }else{
                    _self.newscheck(_self.pro);
                    _self.housecheck(_self.pro,_self.city);
                    //_self.autocheck(_self.pro,_self.city);
            		_self.autoPriceCheck(_self.city);
                    _self.tuancheck(_self.pro,_self.city);
                    _self.meishicheck(_self.pro);
                }
            });
        }else if(customProvince != null && customCity != null){
            _self.weathercheck(customProvince,customCity);
            _self.locationCustom(_self.pro,_self.city,_self.pro,_self.city);
            _self.locationCustomInit(customProvince,customCity);
            if(customNewsProvince != null && customNewsCity != null){
                ipProvince = customNewsProvince;
                _self.newscheck(customNewsProvince);
                //_self.autocheck(customNewsProvince);
            	_self.autoPriceCheck(customNewsCity);
                _self.tuancheck(customNewsProvince,customNewsCity);
                _self.meishicheck(customNewsProvince);
                _self.housecheck(customNewsProvince,customNewsCity);
            }
        }else{
            _self.weathercheck(_self.pro,_self.city);
            _self.locationCustom(_self.pro,_self.city,_self.pro,_self.city);
            _self.locationCustomInit(_self.pro,_self.city);
            _self.newscheck(_self.pro);
            //_self.autocheck(_self.pro);
            _self.autoPriceCheck(_self.city);
            _self.tuancheck(_self.pro,_self.city);
            _self.meishicheck(_self.pro);
            _self.housecheck(_self.pro,_self.city);
        }
    }
}
var login = {
    loginCheck:function(){
        if(QQ.Cookie.get("skey")){
            uin = Number(QQ.Cookie.get("uin").substring(1));
            skey = QQ.Cookie.get("skey");
            try{
                QQ.LoadScript("http://qfwd.qq.com/?uin=" + uin + "&skey=" + skey + "&func=loginAll&refresh=0&ran="+Math.random(), function(){}, "utf-8");
            }catch(e){}
        }else{
            ipMe.init();
            $("#newsMoreBtn a").html("您有未读新闻，点击查看");
            /*
             if(QQ.localData.get("nick")){
             }*/
            QQ.LoadScript("http://mat1.gtimg.com/finance/js/st/p/qq/q_fin_v20140623144932.js?ran="+Math.random(),function(){});
        }
    },
    loginOut:function(){
        var _this = this;
        QQ.LoadScript("https://ui.ptlogin2.qq.com/js/ptloginout.js",function(){
            pt_logout.logout(function(n){
                if(n){
                    _this.loginGrayOutCall();
                    _this.loginOutCall();
                }
            });
        });
    },
    loginGrayOutCall:function(){
        //一键登录灰度订阅重置
        $("#dingyueGrayTitle").html("订阅：");
        DingyueTotalnum = undefined;
        //$("#dingyueGrayNum").hide();
        $("#dingyueGraySmart").hide();
        $("#dingyueGrayLi").hide();
        //一键登录灰度邮箱重置
        $("#mailGrayTitle").html("QQ邮箱：");
        $("#mailGrayNum").hide();
        $("#inboxGrayLi").hide();
        $("#bottleGrayLi").hide();
        $("#gmailGrayLi").hide();
        $("#dmailGrayLi").hide();
        //一键登录灰度空间重置
        $("#qzoneGrayTitle").html("QQ空间：");
        $("#qzoneGrayNum").hide();
        $("#passiveGrayLi").hide();
        $("#InitGrayLi").hide();
        $("#AboutGrayLi").hide();
        //一键登录灰度微博重置
        $("#weiboGrayTitle").html("腾讯微博：");
        $("#weiboGrayNum").hide();
        $("#msgGrayLi").hide();
        $("#atGrayLi").hide();
        $("#fansGrayLi").hide();
        //一键登录灰度登录重置
        $("#loginGraySmart").hide();
        $("#userVipLayout").hide();
        $("#userVipHead").hide();
        $("#loginedGray").hide();
        $("#loginGray").show();
    },
    loginOutCall:function(){
        DingyueTotalnum = undefined;
        QQ.LoadScript("http://mat1.gtimg.com/finance/js/st/p/qq/q_fin_v20140623144932.js?ran="+Math.random(),function(){
            __.app.qindex.pStock.clearCookie();
        });
    },
    loginSuccess:function(){
        document.getElementById("layoutBg").style.display = "none";
        document.getElementById("login_div").style.left = "-9999px";
        $("#loginedGray").show();
        $("#loginGray").hide();
        login.loginCheck();
    }
}

var Tab = function(){ this.initialize.apply(this, arguments);};
Tab.prototype = {
    initialize: function(options, onchange){
        this.onchange = onchange || null;
        this.setOptions(options);
        this.binding();
    },
    setOptions: function(options){
        this.tabElement = options.tabElement || null;
        this.contentElement = options.contentElement || null;
        this.currentClass = options.currentClass || 'selected';
        this.eventType = options.eventType || 'onclick';
        this.defaultIndex = options.defaultIndex || 0;
        this.deferTime = Math.abs(options.deferTime) || 0;
        this.mateAttribute = options.mateAttribute || null;
        this.tabTimer = null;
        this.targetCache = null;
        this.instance = [];
    },
    binding: function(){
        var tabs = this.tabElement.children,
                contents = this.contentElement.children,
                i = tabs.length,
                that = this,
                di = (this.defaultIndex%i+i)%i;

        if( i != contents.length )
            throw new Error("tabs numbers or contents numbers erorr!");

        for(;i--;){
            if(this.mateAttribute){
                var j=contents.length, att = tabs[i].getAttribute(this.mateAttribute);
                while(j--){
                    if(contents[j].getAttribute(this.mateAttribute) == att){
                        tabs[i].setAttribute('index', j);
                        break;
                    }
                }
            }else{
                tabs[i].setAttribute('index', i);
            }
        }

        tabs[di].className = that.currentClass;
        var contentIndex = tabs[di].getAttribute('index');
        contents[contentIndex].style.display = 'block';
        this.targetCache = tabs[di];
        if(that.onchange){
            that.onchange(that, di, tabs[di], contents[contentIndex]);
        }

        this.tabElement[this.eventType] = function(e){
            e = e||window.event;
            var target = e.target||e.srcElement;
            if(this != target.parentNode){ return;}

            window.clearTimeout(that.tabTimer);
            that.tabTimer = window.setTimeout((function(target){
                return function(){
                    var index = target.getAttribute('index');
                    var cacheIndex = that.targetCache.getAttribute('index');
                    if(index != cacheIndex){
                        target.className = that.currentClass;
                        contents[index].style.display = 'block';

                        that.targetCache.className = '';
                        contents[cacheIndex].style.display = 'none';

                        that.targetCache = target;
                        if(that.onchange){ that.onchange(that, index, target, contents[index]);}
                    }
                };
            })(target), that.deferTime);

            if('function' === typeof e.preventDefault){
                e.preventDefault();
                e.stopPropagation();
            }else{
                e.returnValue = false;
                e.cancelBubble = true;
            }
            return false;
        };
        if(that.deferTime>0){
            that.tabElement.onmouseout = function(){
                window.clearTimeout(that.tabTimer);
            };
        }
    }
};
//new Tab({tabElement:document.getElementById('qq-tabs1'), contentElement:document.getElementById('qq-contents1'), eventType:'onmouseover'});
new Tab({tabElement:document.getElementById('qq-tabs2'), contentElement:document.getElementById('qq-contents2'), eventType:'onmouseover'},function(that,index){ //要闻
    if(index == 1){
        bossPuguang(difangTabMap[ipProvince],"tab");
    }
    updateStatus();
});
/*
 new Tab({tabElement:document.getElementById('qq-tabs3'), contentElement:document.getElementById('qq-contents3'), eventType:'onmouseover', deferTime:150});
 var eduNum = document.getElementById('qq-tabs4').children.length;
 var eduRan = Math.floor(Math.random()*eduNum);
 new Tab({tabElement:document.getElementById('qq-tabs4'), contentElement:document.getElementById('qq-contents4'), eventType:'onmouseover', defaultIndex:eduRan}, function(that,index){
 if(index == 0){
 document.getElementById('qq-tabs4').getElementsByTagName("li")[0].style.width = "67px";
 }else{
 document.getElementById('qq-tabs4').getElementsByTagName("li")[0].style.width = "70px";
 }
 });
 */
document.getElementById('jrrb').className = ""; //今日热播
document.getElementById('qq-contents5').getElementsByTagName("div")[0].className = "undis";

/*
 var dateTime = new Date(), index=0;
 */

var dateTime = serverTime, index=0;

var week = dateTime.getDay(), hours = dateTime.getHours();
/*if((week == 5 && hours >= 17) || week == 6 || (week == 0 && hours < 23)){
    index = 2;
}else */
  
//if(dateTime.getHours() >= 18 || dateTime.getHours() < 6){
if(dateTime.getHours() >= 10 && dateTime.getHours() < 22){ 
    index = 1;
}
new Tab({tabElement:document.getElementById('qq-tabs5'), contentElement:document.getElementById('qq-contents5'), eventType:'onmouseover', deferTime:150, defaultIndex:index}, function(that,index){
    if(index == 1){
        defaultPuguang("vartsexp",1787);
    }
});

new Tab({tabElement:document.getElementById('qq-tabs6'), contentElement:document.getElementById('qq-contents6'), eventType:'onmouseover', defaultIndex:1}); //汽车产品库
//new Tab({tabElement:document.getElementById('qq-tabs7'), contentElement:document.getElementById('qq-contents7'), eventType:'onmouseover'}); //汽车旅游

function defaultPuguang(sOp,iTy){
    var userUin = "";
    if(QQ.Cookie.get("o_cookie")){
        userUin = Number(QQ.Cookie.get("o_cookie"));
    }
    var defaultPuguangImage = new Image(1,1);
    defaultPuguangImage.src="http://btrace.qq.com/collect?sIp=&iQQ=" + userUin + "&sBiz=new.qq.com&sOp=" + sOp + "&iSta=&iTy=" + iTy + "&iFlow=&district=&ran=" + Math.random();
}


function bossPuguang(provice,type){
    var userUin = "";
    if(QQ.Cookie.get("o_cookie")){
        userUin = Number(QQ.Cookie.get("o_cookie"));
    }
    var bossPuguangImage = new Image(1,1);
    bossPuguangImage.src="http://btrace.qq.com/collect?sIp=&iQQ=" + userUin + "&sBiz=&sOp=" + type + "&iSta=&iTy=1557&iFlow=&district=" + provice +"&ran=" + Math.random();
}

//产品导航扩展
function productNavMore(){
    var pronavTime = "";
    var productFloatBtnBoolen = 0;
    $("#productNav").bind("mouseover",function(){
        if(pronavTime){
            clearTimeout(pronavTime);
        }
        if($("#productFloatBtn").attr("class") == "productFloatBtn" && productFloatBtnBoolen == 0){
            $("#productFloatBtn").attr("class","productFloatBtnHover");
            $("#productFloatBtn").attr("bossZone","prounford");
            productFloatBtnBoolen = 1;
        }
    });
    $("#productNav").bind("mouseout",function(){
        pronavTime = setTimeout(function(){
            productFloatBtnBoolen = 0;
            $("#productFloat").hide();
            $("#productFloat").css({"width":"0px"});
            $("#productFloatLayout").css({"width":"14px","right":"279px"});
            $("#productFloatBtn").attr("class","productFloatBtn");
        },250);
    });
    $("#productFloatBtn").bind("click",function(){
        if($(this).hasClass("productFloatBtnHover")){
            defaultPuguang("proopen",1787);
            $("#productFloat").show();
            $("#productFloatLayout").css({"width":"364px","right":"278px"});
            $('#productFloat').animate({
                width: 350
            }, 250, function() {
            });
            $("#productFloatBtn").attr("class","productFloatBtnClicked");
        }else if($("#productFloatBtn").hasClass("productFloatBtnClicked")){
            defaultPuguang("profold",1787);
            $('#productFloat').animate({
                width: 0
            }, 250, function() {
            });
            $('#productFloatLayout').animate({
                width: 14
            }, 250, function() {
            });
            $("#productFloatBtn").attr("class","productFloatBtn");
        }
    });
}

//二维码
var erweimaNewsBoolen = true;
function erweimaNews(){
    if($("#erweima")){
        if(document.body.clientWidth >= 1230 && erweimaNewsBoolen){
            $("#erweima").show();
            $("#shutDown").bind("click",function(){
                $("#erweima").hide();
                erweimaNewsBoolen = false;
            });
        }else{
            $("#erweima").hide();
        }
    }
    if($("#erweimaBeta")){
        if(document.body.clientWidth >= 1230 && erweimaNewsBoolen){
            $("#erweimaBeta").show();
            $("#shutDownBeta").bind("click",function(){
                $("#erweimaBeta").hide();
                erweimaNewsBoolen = false;
            });
        }else{
            $("#erweimaBeta").hide();
        }
    }
}
$(window).bind("resize",function(){
    erweimaNews();
});
function erweimaHide(){
    if($("#erweima")){
        $("#erweima").hide();
    }
    if($("#erweimaBeta")){
        $("#erweimaBeta").hide();
    }
    erweimaNewsBoolen = false;
}

function contains(parentNode, childNode) {
    if (parentNode.contains) {
        return parentNode != childNode && parentNode.contains(childNode);
    } else {
        return !!(parentNode.compareDocumentPosition(childNode) & 16);
    }
}
function checkHover(e,target){
    if (getEvent(e).type=="mouseover")  {
        return !contains(target,getEvent(e).relatedTarget||getEvent(e).fromElement) && !((getEvent(e).relatedTarget||getEvent(e).fromElement)===target);
    } else {
        return !contains(target,getEvent(e).relatedTarget||getEvent(e).toElement) && !((getEvent(e).relatedTarget||getEvent(e).toElement)===target);
    }
}
function getEvent(e){
    return e||window.event;
}

//要闻区数据初始化
function newsInit(){
    QQ.LoadScript("http://i.match.qq.com/qhome/ninjayw?val=tubd_qhome_yw&random="+Math.random(),function(){
        if(tubd_qhome_yw.code == 0){
            var tubdYwArr = [];
            for(var i=0; i<8; i++){
                if(tubd_qhome_yw.hot[i].is_locked && tubd_qhome_yw.hot[i].is_locked == "2"){
                    tubdYwArr.push('<li>');
                    for(var n=0; n<tubd_qhome_yw.hot[i].items.length; n++){
                        tubdYwArr.push('<a  class="');
                        if(tubd_qhome_yw.hot[i].items[n].ext){
                            if(tubd_qhome_yw.hot[i].items[n].ext.style.bold == "1"){ //加粗
                                tubdYwArr.push('Q-bold ');
                            }
                            if(tubd_qhome_yw.hot[i].items[n].ext.style.color == "1"){ //红色
                                tubdYwArr.push('qm_c_1 ');
                            }else if(tubd_qhome_yw.hot[i].items[n].ext.style.color == "2"){ //黑色
                                tubdYwArr.push('qm_c_2 ');
                            }else if(tubd_qhome_yw.hot[i].items[n].ext.style.color == "3"){ //蓝色
                                tubdYwArr.push('qm_c_3 ');
                            }
                            if(tubd_qhome_yw.hot[i].items[n].ext.style.icon == "1"){ //微博图标
                                tubdYwArr.push('qm_i_1');
                            }else if(tubd_qhome_yw.hot[i].items[n].ext.style.icon == "2"){ //视频图标
                                tubdYwArr.push('qm_i_2');
                            }else if(tubd_qhome_yw.hot[i].items[n].ext.style.icon == "3"){ //图片图标
                                tubdYwArr.push('qm_i_3');
                            }else if(tubd_qhome_yw.hot[i].items[n].ext.style.icon == "4"){ //直播图标
                                tubdYwArr.push('qm_i_4');
                            }else if(tubd_qhome_yw.hot[i].items[n].ext.style.icon == "5"){ //声音图标
                                tubdYwArr.push('audioico');
                            }
                        }
                        tubdYwArr.push('" target="_blank" href="'+tubd_qhome_yw.hot[i].items[n].url+'">'+tubd_qhome_yw.hot[i].items[n].title+'</a>');
                        if(tubd_qhome_yw.hot[i].items[n].ext){ //竖线
                            if(tubd_qhome_yw.hot[i].items[n].ext.split == "1"){ //竖线
                                tubdYwArr.push(' | ');
                            }else{
                                tubdYwArr.push(' ');
                            }
                        }else{
                            tubdYwArr.push(' ');
                        }
                    }
                    tubdYwArr.push('</li>');
                }else{
                    tubdYwArr.push('<li><a class="');
                    if(tubd_qhome_yw.hot[i].ext){
                        if(tubd_qhome_yw.hot[i].ext.style.bold == "1"){ //加粗
                            tubdYwArr.push('Q-bold ');
                        }
                        if(tubd_qhome_yw.hot[i].ext.style.color == "1"){ //红色
                            tubdYwArr.push('qm_c_1 ');
                        }else if(tubd_qhome_yw.hot[i].ext.style.color == "2"){ //黑色
                            tubdYwArr.push('qm_c_2 ');
                        }else if(tubd_qhome_yw.hot[i].ext.style.color == "3"){ //蓝色
                            tubdYwArr.push('qm_c_3 ');
                        }
                        if(tubd_qhome_yw.hot[i].ext.style.icon == "1"){ //微博图标
                            tubdYwArr.push('qm_i_1');
                        }else if(tubd_qhome_yw.hot[i].ext.style.icon == "2"){ //视频图标
                            tubdYwArr.push('qm_i_2');
                        }else if(tubd_qhome_yw.hot[i].ext.style.icon == "3"){ //图片图标
                            tubdYwArr.push('qm_i_3');
                        }else if(tubd_qhome_yw.hot[i].ext.style.icon == "4"){ //直播图标
                            tubdYwArr.push('qm_i_4');
                        }else if(tubd_qhome_yw.hot[i].ext.style.icon == "5"){ //声音图标
                            tubdYwArr.push('audioico');
                        }
                    }
                    tubdYwArr.push('" target="_blank" href="'+tubd_qhome_yw.hot[i].url+'">'+tubd_qhome_yw.hot[i].title+'</a></li>');
                }
            }
            $("#newsContent01 ul").html(tubdYwArr.join(""));
            $("#newsContent01 ul li:first a").addClass("Q-bold").addClass("qm_c_3");
            tubdYwArr = [];
            $("#newsContent02 .imgArea").html('<a target="_blank" href="'+tubd_qhome_yw.hot[8].url+'"><img src="'+tubd_qhome_yw.hot[8].img+'" alt="'+tubd_qhome_yw.hot[8].title+'" /></a>');
            $("#newsContent02 .txtArea h3").html('<a target="_blank" href="'+tubd_qhome_yw.hot[8].url+'">'+tubd_qhome_yw.hot[8].title+'</a>');
            for(var i=9; i<17; i++){
                if(tubd_qhome_yw.hot[i].is_locked && tubd_qhome_yw.hot[i].is_locked == "2"){
                    tubdYwArr.push('<li>');
                    for(var n=0; n<tubd_qhome_yw.hot[i].items.length; n++){
                        tubdYwArr.push('<a  class="');
                        if(tubd_qhome_yw.hot[i].items[n].ext){
                            if(tubd_qhome_yw.hot[i].items[n].ext.style.bold == "1"){ //加粗
                                tubdYwArr.push('Q-bold ');
                            }
                            if(tubd_qhome_yw.hot[i].items[n].ext.style.color == "1"){ //红色
                                tubdYwArr.push('qm_c_1 ');
                            }else if(tubd_qhome_yw.hot[i].items[n].ext.style.color == "2"){ //黑色
                                tubdYwArr.push('qm_c_2 ');
                            }else if(tubd_qhome_yw.hot[i].items[n].ext.style.color == "3"){ //蓝色
                                tubdYwArr.push('qm_c_3 ');
                            }
                            if(tubd_qhome_yw.hot[i].items[n].ext.style.icon == "1"){ //微博图标
                                tubdYwArr.push('qm_i_1');
                            }else if(tubd_qhome_yw.hot[i].items[n].ext.style.icon == "2"){ //视频图标
                                tubdYwArr.push('qm_i_2');
                            }else if(tubd_qhome_yw.hot[i].items[n].ext.style.icon == "3"){ //图片图标
                                tubdYwArr.push('qm_i_3');
                            }else if(tubd_qhome_yw.hot[i].items[n].ext.style.icon == "4"){ //直播图标
                                tubdYwArr.push('qm_i_4');
                            }else if(tubd_qhome_yw.hot[i].items[n].ext.style.icon == "5"){ //声音图标
                                tubdYwArr.push('audioico');
                            }
                        }
                        tubdYwArr.push('" target="_blank" href="'+tubd_qhome_yw.hot[i].items[n].url+'">'+tubd_qhome_yw.hot[i].items[n].title+'</a>');
                        if(tubd_qhome_yw.hot[i].items[n].ext){ //竖线
                            if(tubd_qhome_yw.hot[i].items[n].ext.split == "1"){ //竖线
                                tubdYwArr.push(' | ');
                            }else{
                                tubdYwArr.push(' ');
                            }
                        }else{
                            tubdYwArr.push(' ');
                        }
                    }
                    tubdYwArr.push('</li>');
                }else{
                    tubdYwArr.push('<li><a class="');
                    if(tubd_qhome_yw.hot[i].ext){
                        if(tubd_qhome_yw.hot[i].ext.style.bold == "1"){ //加粗
                            tubdYwArr.push('Q-bold ');
                        }
                        if(tubd_qhome_yw.hot[i].ext.style.color == "1"){ //红色
                            tubdYwArr.push('qm_c_1 ');
                        }else if(tubd_qhome_yw.hot[i].ext.style.color == "2"){ //黑色
                            tubdYwArr.push('qm_c_2 ');
                        }else if(tubd_qhome_yw.hot[i].ext.style.color == "3"){ //蓝色
                            tubdYwArr.push('qm_c_3 ');
                        }
                        if(tubd_qhome_yw.hot[i].ext.style.icon == "1"){ //微博图标
                            tubdYwArr.push('qm_i_1');
                        }else if(tubd_qhome_yw.hot[i].ext.style.icon == "2"){ //视频图标
                            tubdYwArr.push('qm_i_2');
                        }else if(tubd_qhome_yw.hot[i].ext.style.icon == "3"){ //图片图标
                            tubdYwArr.push('qm_i_3');
                        }else if(tubd_qhome_yw.hot[i].ext.style.icon == "4"){ //直播图标
                            tubdYwArr.push('qm_i_4');
                        }else if(tubd_qhome_yw.hot[i].ext.style.icon == "5"){ //声音图标
                            tubdYwArr.push('audioico');
                        }
                    }
                    tubdYwArr.push('" target="_blank" href="'+tubd_qhome_yw.hot[i].url+'">'+tubd_qhome_yw.hot[i].title+'</a></li>');
                }
            }
            $("#newsContent02 ul").html(tubdYwArr.join(""));
            tubdYwArr = [];
            $("#newsContent03 .imgArea").html('<a target="_blank" href="'+tubd_qhome_yw.hot[17].url+'"><img src="'+tubd_qhome_yw.hot[17].img+'" alt="'+tubd_qhome_yw.hot[17].title+'" /></a>');
            $("#newsContent03 .txtArea h3").html('<a target="_blank" href="'+tubd_qhome_yw.hot[17].url+'">'+tubd_qhome_yw.hot[17].title+'</a>');
            for(var i=0; i<9; i++){
                if(tubd_qhome_yw.gxh[i].is_locked && tubd_qhome_yw.gxh[i].is_locked == "2"){
                    tubdYwArr.push('<li>');
                    for(var n=0; n<tubd_qhome_yw.gxh[i].items.length; n++){
                        tubdYwArr.push('<a  class="');
                        if(tubd_qhome_yw.gxh[i].items[n].ext){
                            if(tubd_qhome_yw.gxh[i].items[n].ext.style.bold == "1"){ //加粗
                                tubdYwArr.push('Q-bold ');
                            }
                            if(tubd_qhome_yw.gxh[i].items[n].ext.style.color == "1"){ //红色
                                tubdYwArr.push('qm_c_1 ');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.color == "2"){ //黑色
                                tubdYwArr.push('qm_c_2 ');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.color == "3"){ //蓝色
                                tubdYwArr.push('qm_c_3 ');
                            }
                            if(tubd_qhome_yw.gxh[i].items[n].ext.style.icon == "1"){ //微博图标
                                tubdYwArr.push('qm_i_1');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.icon == "2"){ //视频图标
                                tubdYwArr.push('qm_i_2');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.icon == "3"){ //图片图标
                                tubdYwArr.push('qm_i_3');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.icon == "4"){ //直播图标
                                tubdYwArr.push('qm_i_4');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.icon == "5"){ //声音图标
                                tubdYwArr.push('audioico');
                            }
                        }
                        tubdYwArr.push('" target="_blank" href="'+tubd_qhome_yw.gxh[i].items[n].url+'">'+tubd_qhome_yw.gxh[i].items[n].title+'</a>');
                        if(tubd_qhome_yw.gxh[i].items[n].ext){ //竖线
                            if(tubd_qhome_yw.gxh[i].items[n].ext.split == "1"){ //竖线
                                tubdYwArr.push(' | ');
                            }else{
                                tubdYwArr.push(' ');
                            }
                        }else{
                            tubdYwArr.push(' ');
                        }
                    }
                    tubdYwArr.push('</li>');
                }else{
                    tubdYwArr.push('<li><a class="');
                    if(tubd_qhome_yw.gxh[i].ext){
                        if(tubd_qhome_yw.gxh[i].ext.style.bold == "1"){ //加粗
                            tubdYwArr.push('Q-bold ');
                        }
                        if(tubd_qhome_yw.gxh[i].ext.style.color == "1"){ //红色
                            tubdYwArr.push('qm_c_1 ');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.color == "2"){ //黑色
                            tubdYwArr.push('qm_c_2 ');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.color == "3"){ //蓝色
                            tubdYwArr.push('qm_c_3 ');
                        }
                        if(tubd_qhome_yw.gxh[i].ext.style.icon == "1"){ //微博图标
                            tubdYwArr.push('qm_i_1');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.icon == "2"){ //视频图标
                            tubdYwArr.push('qm_i_2');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.icon == "3"){ //图片图标
                            tubdYwArr.push('qm_i_3');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.icon == "4"){ //直播图标
                            tubdYwArr.push('qm_i_4');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.icon == "5"){ //声音图标
                            tubdYwArr.push('audioico');
                        }
                    }
                    tubdYwArr.push('" target="_blank" href="'+tubd_qhome_yw.gxh[i].url+'">'+tubd_qhome_yw.gxh[i].title+'</a></li>');
                }
            }
            $("#newsContent03 ul").html(tubdYwArr.join(""));
            tubdYwArr = [];
            $("#newsContent04 .imgArea").html('<a target="_blank" href="'+tubd_qhome_yw.gxh[9].url+'"><img src="'+tubd_qhome_yw.gxh[9].img+'" alt="'+tubd_qhome_yw.gxh[9].title+'" /></a>');
            $("#newsContent04 .txtArea h3").html('<a target="_blank" href="'+tubd_qhome_yw.gxh[9].url+'">'+tubd_qhome_yw.gxh[9].title+'</a>');
            for(var i=10; i<18; i++){
                if(tubd_qhome_yw.gxh[i].is_locked && tubd_qhome_yw.gxh[i].is_locked == "2"){
                    tubdYwArr.push('<li>');
                    for(var n=0; n<tubd_qhome_yw.gxh[i].items.length; n++){
                        tubdYwArr.push('<a  class="');
                        if(tubd_qhome_yw.gxh[i].items[n].ext){
                            if(tubd_qhome_yw.gxh[i].items[n].ext.style.bold == "1"){ //加粗
                                tubdYwArr.push('Q-bold ');
                            }
                            if(tubd_qhome_yw.gxh[i].items[n].ext.style.color == "1"){ //红色
                                tubdYwArr.push('qm_c_1 ');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.color == "2"){ //黑色
                                tubdYwArr.push('qm_c_2 ');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.color == "3"){ //蓝色
                                tubdYwArr.push('qm_c_3 ');
                            }
                            if(tubd_qhome_yw.gxh[i].items[n].ext.style.icon == "1"){ //微博图标
                                tubdYwArr.push('qm_i_1');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.icon == "2"){ //视频图标
                                tubdYwArr.push('qm_i_2');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.icon == "3"){ //图片图标
                                tubdYwArr.push('qm_i_3');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.icon == "4"){ //直播图标
                                tubdYwArr.push('qm_i_4');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.icon == "5"){ //声音图标
                                tubdYwArr.push('audioico');
                            }
                        }
                        tubdYwArr.push('" target="_blank" href="'+tubd_qhome_yw.gxh[i].items[n].url+'">'+tubd_qhome_yw.gxh[i].items[n].title+'</a>');
                        if(tubd_qhome_yw.gxh[i].items[n].ext){ //竖线
                            if(tubd_qhome_yw.gxh[i].items[n].ext.split == "1"){ //竖线
                                tubdYwArr.push(' | ');
                            }else{
                                tubdYwArr.push(' ');
                            }
                        }else{
                            tubdYwArr.push(' ');
                        }
                    }
                    tubdYwArr.push('</li>');
                }else{
                    tubdYwArr.push('<li><a class="');
                    if(tubd_qhome_yw.gxh[i].ext){
                        if(tubd_qhome_yw.gxh[i].ext.style.bold == "1"){ //加粗
                            tubdYwArr.push('Q-bold ');
                        }
                        if(tubd_qhome_yw.gxh[i].ext.style.color == "1"){ //红色
                            tubdYwArr.push('qm_c_1 ');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.color == "2"){ //黑色
                            tubdYwArr.push('qm_c_2 ');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.color == "3"){ //蓝色
                            tubdYwArr.push('qm_c_3 ');
                        }
                        if(tubd_qhome_yw.gxh[i].ext.style.icon == "1"){ //微博图标
                            tubdYwArr.push('qm_i_1');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.icon == "2"){ //视频图标
                            tubdYwArr.push('qm_i_2');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.icon == "3"){ //图片图标
                            tubdYwArr.push('qm_i_3');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.icon == "4"){ //直播图标
                            tubdYwArr.push('qm_i_4');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.icon == "5"){ //声音图标
                            tubdYwArr.push('audioico');
                        }
                    }
                    tubdYwArr.push('" target="_blank" href="'+tubd_qhome_yw.gxh[i].url+'">'+tubd_qhome_yw.gxh[i].title+'</a></li>');
                }
            }
            $("#newsContent04 ul").html(tubdYwArr.join(""));
            tubdYwArr = [];
            $("#newsContent05 .imgArea").html('<a target="_blank" href="'+tubd_qhome_yw.gxh[18].url+'"><img src="'+tubd_qhome_yw.gxh[18].img+'" alt="'+tubd_qhome_yw.gxh[18].title+'" /></a>');
            $("#newsContent05 .txtArea h3").html('<a target="_blank" href="'+tubd_qhome_yw.gxh[18].url+'">'+tubd_qhome_yw.gxh[18].title+'</a>');
            for(var i=19; i<27; i++){
                if(tubd_qhome_yw.gxh[i].is_locked && tubd_qhome_yw.gxh[i].is_locked == "2"){
                    tubdYwArr.push('<li>');
                    for(var n=0; n<tubd_qhome_yw.gxh[i].items.length; n++){
                        tubdYwArr.push('<a  class="');
                        if(tubd_qhome_yw.gxh[i].items[n].ext){
                            if(tubd_qhome_yw.gxh[i].items[n].ext.style.bold == "1"){ //加粗
                                tubdYwArr.push('Q-bold ');
                            }
                            if(tubd_qhome_yw.gxh[i].items[n].ext.style.color == "1"){ //红色
                                tubdYwArr.push('qm_c_1 ');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.color == "2"){ //黑色
                                tubdYwArr.push('qm_c_2 ');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.color == "3"){ //蓝色
                                tubdYwArr.push('qm_c_3 ');
                            }
                            if(tubd_qhome_yw.gxh[i].items[n].ext.style.icon == "1"){ //微博图标
                                tubdYwArr.push('qm_i_1');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.icon == "2"){ //视频图标
                                tubdYwArr.push('qm_i_2');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.icon == "3"){ //图片图标
                                tubdYwArr.push('qm_i_3');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.icon == "4"){ //直播图标
                                tubdYwArr.push('qm_i_4');
                            }else if(tubd_qhome_yw.gxh[i].items[n].ext.style.icon == "5"){ //声音图标
                                tubdYwArr.push('audioico');
                            }
                        }
                        tubdYwArr.push('" target="_blank" href="'+tubd_qhome_yw.gxh[i].items[n].url+'">'+tubd_qhome_yw.gxh[i].items[n].title+'</a>');
                        if(tubd_qhome_yw.gxh[i].items[n].ext){ //竖线
                            if(tubd_qhome_yw.gxh[i].items[n].ext.split == "1"){ //竖线
                                tubdYwArr.push(' | ');
                            }else{
                                tubdYwArr.push(' ');
                            }
                        }else{
                            tubdYwArr.push(' ');
                        }
                    }
                    tubdYwArr.push('</li>');
                }else{
                    tubdYwArr.push('<li><a class="');
                    if(tubd_qhome_yw.gxh[i].ext){
                        if(tubd_qhome_yw.gxh[i].ext.style.bold == "1"){ //加粗
                            tubdYwArr.push('Q-bold ');
                        }
                        if(tubd_qhome_yw.gxh[i].ext.style.color == "1"){ //红色
                            tubdYwArr.push('qm_c_1 ');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.color == "2"){ //黑色
                            tubdYwArr.push('qm_c_2 ');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.color == "3"){ //蓝色
                            tubdYwArr.push('qm_c_3 ');
                        }
                        if(tubd_qhome_yw.gxh[i].ext.style.icon == "1"){ //微博图标
                            tubdYwArr.push('qm_i_1');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.icon == "2"){ //视频图标
                            tubdYwArr.push('qm_i_2');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.icon == "3"){ //图片图标
                            tubdYwArr.push('qm_i_3');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.icon == "4"){ //直播图标
                            tubdYwArr.push('qm_i_4');
                        }else if(tubd_qhome_yw.gxh[i].ext.style.icon == "5"){ //声音图标
                            tubdYwArr.push('audioico');
                        }
                    }
                    tubdYwArr.push('" target="_blank" href="'+tubd_qhome_yw.gxh[i].url+'">'+tubd_qhome_yw.gxh[i].title+'</a></li>');
                }
            }
            $("#newsContent05 ul").html(tubdYwArr.join(""));



            var NewsRemind = {
                init: function(config){
                    var that = this;
                    this.messageBox = config.messageBox;
                    this.upperLimit = config.upperLimit || 12;                // 随机数上限
                    this.lowerLimit = config.lowerLimit || 9;                 // 随机数下限
                    this.maxCount = config.maxCount || 18;                    // 最大新闻量
                    this.isPutTop = config.isPutTop && true;                  // 小黄条是否置顶
                    this.startSecond = (config.startSecond || 20)*1000;       // 多少秒后显示
                    this.holdSecond = config.holdSecond || 3600;              // 点两次后保持的时间，单位秒


                    if(!this.isPutTop){
                        $('#newsContent05').insertBefore('#newsContent03');
                        $('#newsContent04').insertBefore('#newsContent03');
                        $('#newsMoreBtn').insertBefore('#newsContent05');

                        $('#newsContent04 div').attr('bosszone', 'recomheadline3');
                        $('#newsContent04 ul').attr('bosszone', 'recommend3');

                        $('#newsContent05 div').attr('bosszone', 'recomheadline4');
                        $('#newsContent05 ul').attr('bosszone', 'recommend4');
                    }

                    var dateTime = new Date();
                    var lastDay = dateTime.getFullYear() + '-' + (dateTime.getMonth()+1) + '-' + dateTime.getDate();

                    if(lastDay == QQ.localData.get('lastDay')){
                        this.firstCount = parseInt( QQ.localData.get('newsRMS') );
                    }else{
                        QQ.localData.set('lastDay', lastDay);

                        // 生成随机数
                        this.firstCount = parseInt(Math.random()*(that.upperLimit-that.lowerLimit+1)+that.lowerLimit);
                        QQ.localData.set('newsRMS', this.firstCount);
                    }
                    this.twiceCount = 0;


                    window.setTimeout(function(){
                        var lastTS = QQ.localData.get('lastTS'), count=that.firstCount;
                        var isLoadFirst = QQ.localData.get('isLoadFirst');

                        if(isLoadFirst == 'YES'){
                            that.twiceCount = that.maxCount - that.firstCount;
                            count = that.twiceCount;
                        }

                        if(lastTS){
                            var currentTS = parseInt(new Date().getTime()/1000);
                            if(currentTS-parseInt(lastTS) >= that.holdSecond){
                                that.messageBox.innerHTML = that.getInfoByCount( count );
                                $("#newsMoreBtn").show();
                                defaultPuguang("EXnewsmore1",1787);

                                QQ.localData.remove('lastTS');
                            }
                        }else{
                            // 显示小黄条
                            that.messageBox.innerHTML = that.getInfoByCount( count );
                            $("#newsMoreBtn").show();
                            defaultPuguang("EXnewsmore1",1787);
                        }
                    }, that.startSecond);

                    this.messageBox.onclick = function(e){
                        if(this.getElementsByTagName('A').length <= 0){
                            return;
                        }

                        var len = $("#newsContent05 ul")[0].children.length;
                        // 第一次点击20秒后再次显示小黄条
                        if(that.twiceCount == 0){
                            window.setTimeout(function(){
                                // 显示小黄条
                                that.messageBox.innerHTML = that.getInfoByCount( that.twiceCount );
                                $("#newsMoreBtn").show();
                            }, that.startSecond);


                            var count = num = that.firstCount-9;
                            while(count > 0){
                                $("#newsContent04 ul").append(  $("#newsContent05 ul").children().eq(len-num) );
                                count--;
                            }

                            $("#newsContent04").slideDown();

                            QQ.localData.set('isLoadFirst', 'YES');

                            // 数据上报
                            if(that.isPutTop){
                                defaultPuguang("newsmore1",1445);
                            }else{
                                defaultPuguang("newsmore3",1445);
                            }
                        }else{
                            var lastTS = String(parseInt(new Date().getTime()/1000));
                            QQ.localData.set('lastTS', lastTS);

                            if(len > that.twiceCount-1){
                                $("#newsContent05 li:gt(" + (that.twiceCount-2) + ')').remove();
                            }

                            $("#newsContent05").slideDown();


                            QQ.localData.remove('isLoadFirst');

                            // 数据上报
                            if(that.isPutTop){
                                defaultPuguang("newsmore2",1445);
                            }else{
                                defaultPuguang("newsmore4",1445);
                            }
                        }

                        that.twiceCount = that.maxCount - that.firstCount;
                        //$(this).hide();
                        that.messageBox.innerHTML = '暂无更新，休息一会儿';
                    };
                },
                getInfoByCount: function(count){
                    //return '<a href="javascript:void(0)">您有<span>' + count + '</span>条未读新闻，点击查看</a>';
                    return '<a href="javascript:void(0)">您有未读新闻，点击查看</a>';
                },
                showNews: function(count){
                }
            };


            (function(){
                var cookieGet = QQ.Cookie.get;
                var uinStr = cookieGet("uin") || cookieGet("luin") || cookieGet("pt2qquin") || cookieGet("o_cookie") || "";
                uinStr = parseInt(Number(uinStr.replace("o",""))).toString();

                var config = {messageBox:document.getElementById('newsMoreBtn'), isPutTop:true, lowerLimit:9, upperLimit:9, holdSecond:3600};
                /*
                if(uinStr && parseInt(parseInt(uinStr/1000)%10) > 4 ){
                    config.isPutTop = false;
                }*/
                NewsRemind.init(config);
            })();


            /*
             $("#newsMoreBtn").bind("click",function(){
             $("#newsContent04").slideDown();
             $("#newsMoreBtn2").show();
             $(this).hide();

             var lastTS = String(parseInt(new Date().getTime()/1000));
             QQ.localData.set('lastTS', lastTS);
             });
             $("#newsMoreBtn2").bind("click",function(){
             $("#newsContent05").slideDown();
             $("#newsMoreBtn3").show();
             $(this).hide();
             });
             setTimeout(function(){
             var lastTS = QQ.localData.get('lastTS');
             if(lastTS){
             var currentTS = parseInt(new Date().getTime()/1000);
             if(currentTS-parseInt(lastTS) >= 3600){
             $("#newsMoreBtn").show();
             defaultPuguang("EXnewsmore1",1787);

             QQ.localData.remove('lastTS');
             }
             }else{
             $("#newsMoreBtn").show();
             defaultPuguang("EXnewsmore1",1787);
             }
             },20000);
             */
        }
    });
}

var orderCurrentArr = []; //当前板块排列顺序

//上传订制顺序回调
function orderUpdateFun(){

}

//上传订制顺序
function orderUpdate(n){
    if(n == 0){ //清除订制顺序
        var param = encodeURI('{"subobj":{"busi_id":"tubd_sub_23","busi_subid":"","data":[{"k":["none"]}]},"oper":"set"}');
        QQ.LoadScript('http://i.match.qq.com/interface/sub?para=' + param + '&callback=orderUpdateFun&random='+Math.random(),function(){});
    }else{ //获取当前排列顺序并上传
        var orderArr = [];
        for(var i=1; i<5; i++){
            orderArr.push($("#contentLayout0"+i).children().first().attr("id"));
        }
        for(var i=1; i<5; i++){
            orderArr.push($("#contentLayout0"+i).children().eq(1).attr("id"));
        }
        for(var i=5; i<9; i++){
            $("#contentLayout0"+i+" .contentMod").each(function(){
                if($(this).attr("id")){
                    orderArr.push($(this).attr("id"));
                }
            });
        }
        var orderStr = '["' + orderArr.join('","') + '"]';
        var param = encodeURI('{"subobj":{"busi_id":"tubd_sub_23","busi_subid":"","data":[{"k":' + orderStr + '}]},"oper":"set"}');
        QQ.LoadScript('http://i.match.qq.com/interface/sub?para=' + param + '&callback=orderUpdateFun&random='+Math.random(),function(){});
        //http://i.match.qq.com/interface/sub?para={"subobj":{"busi_id":"tubd_sub_23","busi_subid":"","data":[{"k":["ent","astro","fashion","book"]}]},"oper":"set"}
    }
}

//获取订制排序
function customOrder(obj){
    if(obj.code == 0){ //成功返回18个订制顺序
        if(obj.data.length != 0 && obj.data[0].k.length == 18){
            var finalArr = obj.data[0].k; //频道顺序数组
            orderInit(finalArr); //板块排序
            orderCurrentArr = finalArr;
            //添加订制事件
            orderEvent();
        }else{ //拉取用户兴趣频道
            QQ.LoadScript("http://i.match.qq.com/qhome/uinterest?num=4&callback=contentInit&random="+Math.random(),function(){});
        }
    }else{ //拉取用户兴趣频道
        QQ.LoadScript("http://i.match.qq.com/qhome/uinterest?num=4&callback=contentInit&random="+Math.random(),function(){});
    }
}

//获取兴趣频道排序
function contentInit(obj){
    var defaultArr = ["finance","sports","ent","tech","auto","house","book","cul","fashion","dajia","astro","edu","kid","history","games","dy","newssh","rufodao"]; //所有频道数组
    var fixedArr = ["finance","sports","ent"]; //固定展开三大频道
    var contentArr = ["tech","auto","house","book","cul","fashion","dajia","astro","edu","kid","history","games","dy","newssh","rufodao"]; //去除固定展开三大频道数组
    var interestArr = []; //兴趣频道数组
    var interestChanel = ""; //四大频道之外的兴趣频道
    var finalArr = []; //无最高兴趣的最终频道数组

    if(obj.num != "undefined" && obj.num != 0){ //有兴趣频道
        for(var i=0; i<obj.num; i++){ //把兴趣频道放到数组里
            interestArr.push(obj.interest[i].ename);
        }
        for(var i=0; i<interestArr.length; i++){ //判断兴趣频道中是否有四大频道之外的频道
            if(interestArr[i] != "finance" && interestArr[i] != "sports" && interestArr[i] != "ent" && interestArr[i] != "tech"){ //如果有，该兴趣频道就是最高兴趣频道
                interestChanel = interestArr[i];
                if(interestChanel == "comic"){ //动漫兴趣转为游戏
                    interestChanel = "games";
                }
                if(interestChanel == "milite"){ //军事兴趣转为历史
                    interestChanel = "history";
                }
                if(interestChanel == "digi"){ //数码兴趣转为科技
                    interestChanel = "tech";
                }
                for(var n=0; n<contentArr.length; n++){ //把最高兴趣频道从所有频道数组移出
                    if(interestChanel == contentArr[n]){
                        contentArr.splice(n,1);
                        break;
                    }
                }
                break;
            }
        }
        if(interestChanel != ""){ //如果有四大频道之外的兴趣频道
            finalArr = fixedArr.push(interestChanel);
            finalArr = fixedArr.concat(contentArr); //合并兴趣频道
        }else{
            finalArr = ["finance","sports","ent","tech","auto","house","book","cul","fashion","dajia","astro","edu","kid","history","games","dy","newssh","rufodao"];
        }
    }else{ //没有兴趣频道
        finalArr = defaultArr;
    }
    orderInit(finalArr); //板块排序
    orderCurrentArr = finalArr;
    //添加订制事件
    orderEvent();
}

//板块排序
var personalBaoguang = 0;
function orderInit(finalArr){
    $("#contentLayout01").append($("#"+finalArr[0])).append($("#"+finalArr[4]));
    $("#contentLayout02").append($("#"+finalArr[1])).append($("#"+finalArr[5]));
    $("#contentLayout03").append($("#"+finalArr[2])).append($("#"+finalArr[6]));
    $("#contentLayout04").append($("#"+finalArr[3])).append($("#"+finalArr[7]));
    $("#contentLayout05").append($("#"+finalArr[8])).append($("#"+finalArr[9])).append($("#"+finalArr[10]));
    $("#contentLayout06").prepend($("#"+finalArr[12])).prepend($("#"+finalArr[11]));
    $("#contentLayout07").append($("#"+finalArr[13])).append($("#"+finalArr[14])).append($("#"+finalArr[15]));
    $("#contentLayout08").prepend($("#"+finalArr[17])).prepend($("#"+finalArr[16]));
    //设置所有频道为收起状态
    $(".contentMod").removeClass("contentModBig");
    //隐藏所有频道的扩展部分
    $(".contentMod").find(".bd .contentRight").hide()
    //设置4个频道展开
    $("#"+finalArr[0]).addClass("contentModBig");
    $("#"+finalArr[0]).find(".bd .contentRight").show();
    if($("#"+finalArr[0]).find(".bd").attr("bossZone").indexOf("_1") < 0){
        $("#"+finalArr[0]).find(".bd").attr("bossZone",$("#"+finalArr[0]).find(".bd").attr("bossZone")+"_1");
    }
    $("#"+finalArr[1]).addClass("contentModBig");
    $("#"+finalArr[1]).find(".bd .contentRight").show();
    if($("#"+finalArr[1]).find(".bd").attr("bossZone").indexOf("_1") < 0){
        $("#"+finalArr[1]).find(".bd").attr("bossZone",$("#"+finalArr[1]).find(".bd").attr("bossZone")+"_1");
    }
    $("#"+finalArr[2]).addClass("contentModBig");
    $("#"+finalArr[2]).find(".bd .contentRight").show();
    if($("#"+finalArr[2]).find(".bd").attr("bossZone").indexOf("_1") < 0){
        $("#"+finalArr[2]).find(".bd").attr("bossZone",$("#"+finalArr[2]).find(".bd").attr("bossZone")+"_1");
    }
    $("#"+finalArr[3]).addClass("contentModBig");
    $("#"+finalArr[3]).find(".bd .contentRight").show();
    if($("#"+finalArr[3]).find(".bd").attr("bossZone").indexOf("_1") < 0){
        $("#"+finalArr[3]).find(".bd").attr("bossZone",$("#"+finalArr[3]).find(".bd").attr("bossZone")+"_1");
    }
    //设置旅游板块bossZone
    /*
     if($("#auto").hasClass("contentModBig")){
     $("#tripBd").attr("bossZone","trip_1");
     }else{
     $("#tripBd").attr("bossZone","trip");
     }
     */
    //清除每个通栏最右侧模块的右边距
    $(".contentLayout").each(function(){
        $(this).children().css({"margin-right":"20px"});
        $(this).children().last().css({"margin-right":"0"});
    });
    if(personalBaoguang == 0){
        $(".contentMod").each(function(){
            if($(this).find(".bd:first").attr("bosszone")){
                defaultPuguang($(this).find(".bd:first").attr("bosszone"),2128);
            }
        });
        personalBaoguang = 1;
    }
    //娱乐个性化灰度
    entPersonal();
    //娱乐个性化灰度
    fashionPersonal();
}

/**
 * 栏目定制
 * @type {Object}
 * @author harleywang@tencent.com
 */
var ColTailor = {
    config: [{id:'finance', name:'财经'}, {id:'sports', name:'体育'},
        {id:'ent', name:'娱乐'},  {id:'tech', name:'科技'},
        {id:'auto', name:'汽车'}, {id:'house', name:'房产'},
        {id:'fashion', name:'时尚'}, {id:'dajia', name:'大家'},
        {id:'book', name:'读书'}, {id:'edu', name:'教育'},
        {id:'kid', name:'儿童'}, {id:'cul', name:'文化'},
        {id:'history', name:'军事'}, {id:'games', name:'游戏'},
        {id:'newssh', name:'社会'}, {id:'rufodao', name:'儒佛道'},
        {id:'astro', name:'星座'}, {id:'dy', name:'订阅'}],
    onClose: function(target){
        $(target).parents(".orderContent").removeClass("orderContentRight");
        $(target).parents(".orderContent").hide();
        $(target).parents(".orderLayout").hide();
        $(target).parents(".hd").removeClass("hdHover");

        defaultPuguang("cs_closebutton",1445);
    },
    onReset: function(target){
        if(QQ.Cookie.get("skey")){ //判断是否登录
            $(target).parents(".orderContent").removeClass("orderContentRight");
            $(target).parents(".orderContent").hide();
            $(target).parents(".orderLayout").hide();
            $(target).parents(".hd").removeClass("hdHover");
            QQ.LoadScript("http://i.match.qq.com/qhome/uinterest?callback=contentInit&random="+Math.random(),function(){});
            orderUpdate(0);
        }else{
            userLogin();
        }
        defaultPuguang("cs_defult",1445);
    }
};

//订制事件
function orderEvent(){
    var orderArr = ["finance","sports","ent","tech","auto","house","book","cul","fashion","dajia","astro","edu","kid","history","games","dy","newssh","rufodao"];
    var orderTempArr = orderCurrentArr;
    $(".contentMod").each(function(){
        $(this).bind("mouseover",function(){
            $(this).find(".hd").addClass("hdHover");
            $(this).find(".orderLayout").show();
        });
        $(this).bind("mouseleave",function(e){
            $(this).find(".hd").removeClass("hdHover");
            $(this).find(".orderLayout").hide();
            $(this).find(".orderContent").removeClass("orderContentRight");
            $(this).find(".orderContent").hide();
        });
        $(this).find(".orderLayout").bind("mouseleave",function(){
            $(this).find(".orderContent").hide();
        });
        $(this).find(".orderLayout").bind("mouseenter",function(){
            var box = $(this);
            window.setTimeout(function(){
                defaultPuguang("channelswitch",1445);
            }, 10);

            if( box.find(".orderContent").length > 0 ){
                box.find(".orderContent").show();
                return;
            }
            var id = box.parent().parent().attr('id');
            var selects = [], config=ColTailor.config;
            for(var i=0, len=config.length; i<len; i++){
                if(id == config[i].id){
                    selects.push('<li name="' + config[i].id + '"><span class="s">' + config[i].name + '</span></li>');
                }else{
                    selects.push('<li name="' + config[i].id + '" bosszone="cs_' + config[i].id + '"><a href="javascript:void(0)">' + config[i].name + '</a></li>');
                }
            }
            var orderContent = document.createElement('div');
            orderContent.className = 'orderContent';
            orderContent.style.display = 'block';
            orderContent.innerHTML = '<div class="orderhd"><h3>选择替换当前频道</h3></div>\
                            <div class="orderShut" onclick="ColTailor.onClose(this)"></div>\
                            <div class="orderbd">\
                                <ul>' + selects.join('') + '</ul>\
                                <p class="orderReset" onclick="ColTailor.onReset(this)">恢复成默认</p>\
                            </div>';
            box.append(orderContent);

            if(box.parent().parent().attr("id") == box.parent().parent().parent().children().last().attr("id")){ //判断板块是否在最右侧
                box.find(".orderContent").addClass("orderContentRight");
            }
        });
    });
    // 频道订制委托事件
    $('.orderLayout').on('click', 'li', function(){
        if($(this).children().first().hasClass("s")){
        }else{
            if(QQ.Cookie.get("skey")){ //判断是否登录
                var turnBefore = 0, turnAfter = 0; //当前板块和要交换板块在数组中的索引值
                for(var n=0, len=orderTempArr.length; n<len; n++){
                    if(orderTempArr[n] == $(this).parents(".contentMod").attr("id")){
                        turnBefore = n;
                    }else if(orderTempArr[n] == $(this).attr("name")){
                        turnAfter = n;
                    }
                }
                //交换两个板块在数组中的位置
                orderTempArr[turnBefore] = $(this).attr("name");
                orderTempArr[turnAfter] = $(this).parents(".contentMod").attr("id");
                //重新排序板块
                orderInit(orderTempArr);
                //上传订制顺序
                orderUpdate();
            }else{
                userLogin();
            }
            $(this).parents(".orderContent").hide();
            $(this).parents(".orderLayout").hide();
            $(this).parents(".hdHover").removeClass("hdHover");
        }
        //return false;
    });
}

//房产定向
function houseIpInit(){
    $("#houseSelectControl").bind("mouseover",function(){
        $("#houseH2").addClass("spread");
        $("#houseSelect").show();
    });
    $("#houseSelectControl").bind("mouseout",function(){
        $("#houseH2").removeClass("spread");
        $("#houseSelect").hide();
    });
    $("#houseSelect li").bind("click",function(){
        ipMe.housecheck($("#"+this).attr("houseProvince"),$("#"+this).attr("houseCity"));
        $("#houseH2 a").html($("#"+this).attr("houseTitle"));
        $("#houseH2 a").attr("href",$("#"+this).attr("houseLink"));
    });
}

//视觉焦点滚动脚本
function picFocusInit(){
    //$("#picFocusUl li:last").html($("#picFocusUl li:first").html());
    $(".picFocus").bind("mouseover",function(){
        $("#visualup").removeClass("default");
        $("#visualdown").removeClass("default");
    });
    $(".picFocus").bind("mouseout",function(){
        $("#visualup").addClass("default");
        $("#visualdown").addClass("default");
    });
    var lock=false;
    $("#visualup").bind("click",function(){
        if(lock){
            return;
        }
        lock = true;
        var picFocusNum = Number($("#picFocusUl").css("left").replace("px","")) + 1008;
        if(picFocusNum > 0){
            picFocusNum -= 3024;
        }
        $('#picFocusUl').animate({
            left: picFocusNum
        }, 500, function() {
            $(".picFocus .hd .btn").children().removeClass("current");
            $(".picFocus .hd .btn").children().eq(Math.abs(picFocusNum/1008)).addClass("current");
            lock = false;
        });
    });
    $("#visualdown").bind("click",function(){
        if(lock){
            return;
        }
        lock = true;
        var picFocusNum = Number($("#picFocusUl").css("left").replace("px","")) - 1008;
        if(picFocusNum < -2016){
            picFocusNum += 3024;
        }
        $('#picFocusUl').animate({
            left: picFocusNum
        }, 500, function() {
            $(".picFocus .hd .btn").children().removeClass("current");
            $(".picFocus .hd .btn").children().eq(Math.abs(picFocusNum/1008)).addClass("current");
            lock = false;
        });
    });
    $("#visual .btn div").bind("click",function(){
        if(lock){
            return;
        }
        lock = true;
        that = $(this);
        $('#picFocusUl').animate({
            left: -1008*$(this).index()
        }, 500, function() {
            $(".picFocus .hd .btn div").removeClass("current");
            that.addClass("current");
            defaultPuguang("change",1445);
            lock = false;
        });
    });
}
//设为首页
function homeSet(){
    var browserName=navigator.userAgent.toLowerCase();
    if(/theworld/i.test(browserName)){
        document.getElementById("homeSet").href += "#theworld";
    }else if(/lbbrowser/i.test(browserName)){
        document.getElementById("homeSet").href += "#lbbrowser";
    }else if(/bidubrowser/i.test(browserName)){
        document.getElementById("homeSet").href += "#bidubrowser";
    }else if(/se 2.x/i.test(browserName)){
        document.getElementById("homeSet").href += "#sougou";
    }else if(/360se/i.test(browserName)){
        document.getElementById("homeSet").href += "#browser360";
    }else if(/qqbrowser/i.test(browserName)){
        document.getElementById("homeSet").href += "#qqbrowser";
    }else if(/maxthon/i.test(browserName)){
        document.getElementById("homeSet").href += "#aoyou";
    }else if(/msie/i.test(browserName) && !/opera/.test(browserName)){
        document.getElementById("homeSet").href += "#ie";
    }else if(/firefox/i.test(browserName)){
        document.getElementById("homeSet").href += "#firefox";
    }else if(/opera/i.test(browserName)){
        document.getElementById("homeSet").href += "#opera";
    }else if(/webkit/i.test(browserName) &&!(/chrome/i.test(browserName) && /webkit/i.test(browserName) && /mozilla/i.test(browserName))){
        document.getElementById("homeSet").href += "#safari";
    }else if(/chrome/i.test(browserName) && /webkit/i.test(browserName) && /mozilla/i.test(browserName)){
        document.getElementById("homeSet").href += "#chrome";
    }
}
//搜狗脚本
function sogouInit(){
    /*
     $("#searchBtn").bind("mouseover",function(){
     $(this).addClass("searchBtnHover");
     });
     $("#searchBtn").bind("mouseout",function(){
     $(this).removeClass("searchBtnHover");
     });
     */
    QQ.LoadScript("http://mat1.gtimg.com/www/js/qq2012/suggestion_1.1.6_sogou.js",function(){
        var frm = document.forms['soso_search_box'];
        sososmart.init(frm,true,"sogou-wsse-3f7bcd0b3ea822680001"); //第一参数表单对象,第二个参数 点击自动提示元素是否自动搜索，第三个参数 搜狗接口统计参数
      //sogouGray();
    });
}
//平台检测
function platformCheck(){
    if(navigator.platform == "Mac68K" || navigator.platform == "MacPPC" || navigator.platform == "Macintosh" || navigator.platform == "MacIntel"){ //mac系统
        $("#select").addClass("macSelect");
    }
}
//字体适配
function fontCheck(){
    if(navigator.platform == "Win32" || navigator.platform == "Windows"){ //windows系统
        if($("#guess").width() != 86){ //没有微软雅黑字体
            $("body").addClass("noYahei");
        }
    }
    if(navigator.platform == "Mac68K" || navigator.platform == "MacPPC" || navigator.platform == "Macintosh" || navigator.platform == "MacIntel"){ //mac系统
        $("body").addClass("mac");
    }
}
//字体切换
function fontTurn(){
    if(window.location.toString().indexOf('type=song') != -1){ //切换成宋体
        $("body").addClass("song");
        $("#diaocha").find("a").attr("href","http://exp.qq.com/ur/?urid=12565");
    }
}
/*返回顶部*/
function toTopHide(){
    if(document.documentElement.scrollTop+document.body.scrollTop > 400){
        if(!(/AppleWebKit.*Mobile/i.test(navigator.userAgent) || (/MIDP|SymbianOS|NOKIA|SAMSUNG|LG|NEC|TCL|Alcatel|BIRD|DBTEL|Dopod|PHILIPS|HAIER|LENOVO|MOT-|Nokia|SonyEricsson|SIE-|Amoi|ZTE/.test(navigator.userAgent)))){
          //document.getElementById("jieshao").style.display = "block";
            document.getElementById("fankui").style.display = "block";
            document.getElementById("homeSet").style.display = "block";
            document.getElementById("toTop").style.display = "block";
        }
    }else{
      //document.getElementById("jieshao").style.display = "none";
        document.getElementById("fankui").style.display = "none";
        document.getElementById("homeSet").style.display = "none";
        document.getElementById("toTop").style.display = "none";
    }
}
/*移动设备隐藏自选股浮层*/
function stockMobile(){
    if(/Android|Windows Phone|webOS|iPhone|iPod|BlackBerry|iPad/i.test(navigator.userAgent)){
        $("#top-smartbox-du-sx-result").hide();
        $("#stocksearch").hide();
    }
}
/*房产产品库smartbox*/
function houseProduct(){
    if($("#smartbox_form_1")){
        QQ.LoadScript("http://mat1.gtimg.com/www/js/qq2012/smartboxRewrite1.7.js",function(){
            $('#smartbox_form_1').empty();
            createSmartbox("smartbox_form_1");
        });
    }
}
/*教育产品库*/
function getid(v){return document.getElementById(v)};

//提交查询的
function goQuery(){
    var alertMsg = ['请输入院校名称','请输入院校名称'];
    var selectItem = getid("select").selectedIndex;
    if (getid("txt").value == ''||getid("txt").value == '院校名称' ||getid("txt").value == '院校中/英文名称' ){
        alert(alertMsg[selectItem]);
        getid("txt").focus();
        return false;
    }else if (getid("txt").value.length > 200){
        alert(alertMsg[selectItem]+"太长");
        getid("txt").focus();
        return false;
    }

    if (selectItem == 0){
        var queryTemp = getid("txt").value;
        if (/gecko/.test(window.navigator.userAgent.toLowerCase())){
            queryTemp = encodeURI(queryTemp);
        }
        window.open(getid("select").value + "?query=" + queryTemp + "&page=1");
    }else if(selectItem == 1){
        var queryTemp = getid("txt").value;
        if (/gecko/.test(window.navigator.userAgent.toLowerCase())){
            queryTemp = encodeURI(queryTemp);
        }
        window.open(getid("select").value + "?school=" + queryTemp);
    }else{
        getid("seach").submit();
    }
    return false;
}

//初始化输入框-获得焦点的
function focusVerify(){
    var alertMsg = ['院校名称','院校中/英文名称'];
    var selectItem = getid("select").selectedIndex;
    if (getid("txt").value == alertMsg[selectItem]){
        //是默认值，将其值空
        getid("txt").value = '';
    }
}

//初始化输入框-失去焦点的
function blurVerify(){
    var alertMsg = ['院校名称','院校中/英文名称'];
    var selectItem = getid("select").selectedIndex;
    if (getid("txt").value == '')
    {
        //是默认值，将其值空
        getid("txt").value = alertMsg[selectItem];
    }
}
function ints(n){
    var op = getid("select").getElementsByTagName("option");
    switch (n){

        case 0:
            getid("txt").value = "院校名称";
            getid("txt").name="query";
            break;
        case 1:
            getid("txt").value = "院校中/英文名称";
            getid("txt").name="query";
            break;
    }
    for(var i=0;i<op.length;i++){
        if(getid("select").value==op[i].value){
            getid("seach").action =op[i].value;
        }
    }
}
/*汽车产品库*/
var g_boss_img = new Image(1,1);
function BossLinkAutoSearch(iLinkType,iSelect,sKeyword){
    //汽车首页搜索
    g_boss_img.src="http://btrace.qq.com/collect?sIp=&iQQ=&sBiz=&sOp=&iSta=&iTy=259&iFlow=&iPage=1&iLinkType=" + iLinkType + "&iSelect=" + iSelect + "&sKeyword="+sKeyword;
}

var PinYin = {"a":"\u554a\u963f\u9515","ai":"\u57c3\u6328\u54ce\u5509\u54c0\u7691\u764c\u853c\u77ee\u827e\u788d\u7231\u9698\u8bf6\u6371\u55f3\u55cc\u5ad2\u7477\u66a7\u7839\u953f\u972d","an":"\u978d\u6c28\u5b89\u4ffa\u6309\u6697\u5cb8\u80fa\u6848\u8c19\u57ef\u63de\u72b4\u5eb5\u6849\u94f5\u9e4c\u9878\u9eef","ang":"\u80ae\u6602\u76ce","ao":"\u51f9\u6556\u71ac\u7ff1\u8884\u50b2\u5965\u61ca\u6fb3\u5773\u62d7\u55f7\u5662\u5c99\u5ed2\u9068\u5aaa\u9a9c\u8071\u87af\u93ca\u9ccc\u93d6","ba":"\u82ad\u634c\u6252\u53ed\u5427\u7b06\u516b\u75a4\u5df4\u62d4\u8dcb\u9776\u628a\u8019\u575d\u9738\u7f62\u7238\u8307\u83dd\u8406\u636d\u5c9c\u705e\u6777\u94af\u7c91\u9c85\u9b43","bai":"\u767d\u67cf\u767e\u6446\u4f70\u8d25\u62dc\u7a17\u859c\u63b0\u97b4","ban":"\u6591\u73ed\u642c\u6273\u822c\u9881\u677f\u7248\u626e\u62cc\u4f34\u74e3\u534a\u529e\u7eca\u962a\u5742\u8c73\u94a3\u7622\u764d\u8228","bang":"\u90a6\u5e2e\u6886\u699c\u8180\u7ed1\u68d2\u78c5\u868c\u9551\u508d\u8c24\u84a1\u8783","bao":"\u82de\u80de\u5305\u8912\u96f9\u4fdd\u5821\u9971\u5b9d\u62b1\u62a5\u66b4\u8c79\u9c8d\u7206\u52f9\u8446\u5b80\u5b62\u7172\u9e28\u8913\u8db5\u9f85","bo":"\u5265\u8584\u73bb\u83e0\u64ad\u62e8\u94b5\u6ce2\u535a\u52c3\u640f\u94c2\u7b94\u4f2f\u5e1b\u8236\u8116\u818a\u6e24\u6cca\u9a73\u4eb3\u8543\u5575\u997d\u6a97\u64d8\u7934\u94b9\u9e41\u7c38\u8ddb","bei":"\u676f\u7891\u60b2\u5351\u5317\u8f88\u80cc\u8d1d\u94a1\u500d\u72c8\u5907\u60eb\u7119\u88ab\u5b5b\u9642\u90b6\u57e4\u84d3\u5457\u602b\u6096\u789a\u9e4e\u8919\u943e","ben":"\u5954\u82ef\u672c\u7b28\u755a\u574c\u951b","beng":"\u5d29\u7ef7\u752d\u6cf5\u8e66\u8ff8\u552a\u5623\u750f","bi":"\u903c\u9f3b\u6bd4\u9119\u7b14\u5f7c\u78a7\u84d6\u853d\u6bd5\u6bd9\u6bd6\u5e01\u5e87\u75f9\u95ed\u655d\u5f0a\u5fc5\u8f9f\u58c1\u81c2\u907f\u965b\u5315\u4ef3\u4ffe\u8298\u835c\u8378\u5421\u54d4\u72f4\u5eb3\u610e\u6ed7\u6fde\u5f3c\u59a3\u5a62\u5b16\u74a7\u8d32\u7540\u94cb\u79d5\u88e8\u7b5a\u7b85\u7be6\u822d\u895e\u8df8\u9ac0","bian":"\u97ad\u8fb9\u7f16\u8d2c\u6241\u4fbf\u53d8\u535e\u8fa8\u8fa9\u8fab\u904d\u533e\u5f01\u82c4\u5fed\u6c74\u7f0f\u7178\u782d\u78a5\u7a39\u7a86\u8759\u7b3e\u9cca","biao":"\u6807\u5f6a\u8198\u8868\u5a4a\u9aa0\u98d1\u98d9\u98da\u706c\u9556\u9573\u762d\u88f1\u9cd4","bie":"\u9cd6\u618b\u522b\u762a\u8e69\u9cd8","bin":"\u5f6c\u658c\u6fd2\u6ee8\u5bbe\u6448\u50a7\u6d5c\u7f24\u73a2\u6ba1\u8191\u9554\u9acc\u9b13","bing":"\u5175\u51b0\u67c4\u4e19\u79c9\u997c\u70b3\u75c5\u5e76\u7980\u90b4\u6452\u7ee0\u678b\u69df\u71f9","bu":"\u6355\u535c\u54fa\u8865\u57e0\u4e0d\u5e03\u6b65\u7c3f\u90e8\u6016\u62ca\u535f\u900b\u74ff\u6661\u949a\u91ad","ca":"\u64e6\u5693\u7924","cai":"\u731c\u88c1\u6750\u624d\u8d22\u776c\u8e29\u91c7\u5f69\u83dc\u8521","can":"\u9910\u53c2\u8695\u6b8b\u60ed\u60e8\u707f\u9a96\u74a8\u7cb2\u9eea","cang":"\u82cd\u8231\u4ed3\u6ca7\u85cf\u4f27","cao":"\u64cd\u7cd9\u69fd\u66f9\u8349\u8279\u5608\u6f15\u87ac\u825a","ce":"\u5395\u7b56\u4fa7\u518c\u6d4b\u5202\u5e3b\u607b","ceng":"\u5c42\u8e6d\u564c","cha":"\u63d2\u53c9\u832c\u8336\u67e5\u78b4\u643d\u5bdf\u5c94\u5dee\u8be7\u7339\u9987\u6c4a\u59f9\u6748\u6942\u69ce\u6aab\u9497\u9538\u9572\u8869","chai":"\u62c6\u67f4\u8c7a\u4faa\u8308\u7625\u867f\u9f87","chan":"\u6400\u63ba\u8749\u998b\u8c17\u7f20\u94f2\u4ea7\u9610\u98a4\u5181\u8c04\u8c36\u8487\u5edb\u5fcf\u6f7a\u6fb6\u5b71\u7fbc\u5a75\u5b17\u9aa3\u89c7\u7985\u9561\u88e3\u87fe\u8e94","chang":"\u660c\u7316\u573a\u5c1d\u5e38\u957f\u507f\u80a0\u5382\u655e\u7545\u5531\u5021\u4f25\u9b2f\u82cc\u83d6\u5f9c\u6005\u60dd\u960a\u5a3c\u5ae6\u6636\u6c05\u9cb3","chao":"\u8d85\u6284\u949e\u671d\u5632\u6f6e\u5de2\u5435\u7092\u600a\u7ec9\u6641\u8016","che":"\u8f66\u626f\u64a4\u63a3\u5f7b\u6f88\u577c\u5c6e\u7817","chen":"\u90f4\u81e3\u8fb0\u5c18\u6668\u5ff1\u6c89\u9648\u8d81\u886c\u79f0\u8c0c\u62bb\u55d4\u5bb8\u741b\u6987\u809c\u80c2\u789c\u9f80","cheng":"\u6491\u57ce\u6a59\u6210\u5448\u4e58\u7a0b\u60e9\u6f84\u8bda\u627f\u901e\u9a8b\u79e4\u57d5\u5d4a\u5fb5\u6d48\u67a8\u67fd\u6a18\u665f\u584d\u77a0\u94d6\u88ce\u86cf\u9172","chi":"\u5403\u75f4\u6301\u5319\u6c60\u8fdf\u5f1b\u9a70\u803b\u9f7f\u4f88\u5c3a\u8d64\u7fc5\u65a5\u70bd\u50ba\u5880\u82aa\u830c\u640b\u53f1\u54e7\u557b\u55e4\u5f73\u996c\u6cb2\u5ab8\u6555\u80dd\u7719\u7735\u9e31\u761b\u892b\u86a9\u87ad\u7b1e\u7bea\u8c49\u8e05\u8e1f\u9b51","chong":"\u5145\u51b2\u866b\u5d07\u5ba0\u833a\u5fe1\u61a7\u94f3\u825f","chou":"\u62bd\u916c\u7574\u8e0c\u7a20\u6101\u7b79\u4ec7\u7ef8\u7785\u4e11\u4fe6\u5733\u5e31\u60c6\u6eb4\u59af\u7633\u96e0\u9c8b","chu":"\u81ed\u521d\u51fa\u6a71\u53a8\u8e87\u9504\u96cf\u6ec1\u9664\u695a\u7840\u50a8\u77d7\u6410\u89e6\u5904\u4e8d\u520d\u61b7\u7ecc\u6775\u696e\u6a17\u870d\u8e70\u9edc","chuan":"\u63e3\u5ddd\u7a7f\u693d\u4f20\u8239\u5598\u4e32\u63be\u821b\u60f4\u9044\u5ddb\u6c1a\u948f\u9569\u8221","chuang":"\u75ae\u7a97\u5e62\u5e8a\u95ef\u521b\u6006","chui":"\u5439\u708a\u6376\u9524\u5782\u9672\u68f0\u69cc","chun":"\u6625\u693f\u9187\u5507\u6df3\u7eaf\u8822\u4fc3\u83bc\u6c8c\u80ab\u6710\u9e51\u877d","chuo":"\u6233\u7ef0\u851f\u8fb6\u8f8d\u955e\u8e14\u9f8a","ci":"\u75b5\u8328\u78c1\u96cc\u8f9e\u6148\u74f7\u8bcd\u6b64\u523a\u8d50\u6b21\u8360\u5472\u5d6f\u9e5a\u8785\u7ccd\u8d91","cong":"\u806a\u8471\u56f1\u5306\u4ece\u4e1b\u506c\u82c1\u6dd9\u9aa2\u742e\u7481\u679e","cu":"\u51d1\u7c97\u918b\u7c07\u731d\u6b82\u8e59","cuan":"\u8e7f\u7be1\u7a9c\u6c46\u64ba\u6615\u7228","cui":"\u6467\u5d14\u50ac\u8106\u7601\u7cb9\u6dec\u7fe0\u8403\u60b4\u7480\u69b1\u96b9","cun":"\u6751\u5b58\u5bf8\u78cb\u5fd6\u76b4","cuo":"\u64ae\u6413\u63aa\u632b\u9519\u539d\u811e\u9509\u77ec\u75e4\u9e7e\u8e49\u8e9c","da":"\u642d\u8fbe\u7b54\u7629\u6253\u5927\u8037\u54d2\u55d2\u601b\u59b2\u75b8\u8921\u7b2a\u977c\u9791","dai":"\u5446\u6b79\u50a3\u6234\u5e26\u6b86\u4ee3\u8d37\u888b\u5f85\u902e\u6020\u57ed\u7519\u5454\u5cb1\u8fe8\u902f\u9a80\u7ed0\u73b3\u9edb","dan":"\u803d\u62c5\u4e39\u5355\u90f8\u63b8\u80c6\u65e6\u6c2e\u4f46\u60ee\u6de1\u8bde\u5f39\u86cb\u4ebb\u510b\u5369\u840f\u5556\u6fb9\u6a90\u6b9a\u8d55\u7708\u7605\u8043\u7baa","dang":"\u5f53\u6321\u515a\u8361\u6863\u8c20\u51fc\u83ea\u5b95\u7800\u94db\u88c6","dao":"\u5200\u6363\u8e48\u5012\u5c9b\u7977\u5bfc\u5230\u7a3b\u60bc\u9053\u76d7\u53e8\u5541\u5fc9\u6d2e\u6c18\u7118\u5fd1\u7e9b","de":"\u5fb7\u5f97\u7684\u951d","deng":"\u8e6c\u706f\u767b\u7b49\u77aa\u51f3\u9093\u5654\u5d9d\u6225\u78f4\u956b\u7c26","di":"\u5824\u4f4e\u6ef4\u8fea\u654c\u7b1b\u72c4\u6da4\u7fdf\u5ae1\u62b5\u5e95\u5730\u8482\u7b2c\u5e1d\u5f1f\u9012\u7f14\u6c10\u7c74\u8bcb\u8c1b\u90b8\u577b\u839c\u837b\u5600\u5a23\u67e2\u68e3\u89cc\u7825\u78b2\u7747\u955d\u7f9d\u9ab6","dian":"\u98a0\u6382\u6ec7\u7898\u70b9\u5178\u975b\u57ab\u7535\u4f43\u7538\u5e97\u60e6\u5960\u6dc0\u6bbf\u4e36\u963d\u576b\u57dd\u5dc5\u73b7\u765c\u766b\u7c1f\u8e2e","diao":"\u7889\u53fc\u96d5\u51cb\u5201\u6389\u540a\u9493\u8c03\u8f7a\u94de\u8729\u7c9c\u8c82","die":"\u8dcc\u7239\u789f\u8776\u8fed\u8c0d\u53e0\u4f5a\u57a4\u581e\u63f2\u558b\u6e2b\u8f76\u7252\u74de\u8936\u800b\u8e40\u9cbd\u9cce","ding":"\u4e01\u76ef\u53ee\u9489\u9876\u9f0e\u952d\u5b9a\u8ba2\u4e22\u4ec3\u5576\u738e\u815a\u7887\u753a\u94e4\u7594\u8035\u914a","dong":"\u4e1c\u51ac\u8463\u61c2\u52a8\u680b\u4f97\u606b\u51bb\u6d1e\u578c\u549a\u5cbd\u5cd2\u5902\u6c21\u80e8\u80f4\u7850\u9e2b","dou":"\u515c\u6296\u6597\u9661\u8c46\u9017\u75d8\u8538\u94ad\u7aa6\u7aac\u86aa\u7bfc\u9161","du":"\u90fd\u7763\u6bd2\u728a\u72ec\u8bfb\u5835\u7779\u8d4c\u675c\u9540\u809a\u5ea6\u6e21\u5992\u828f\u561f\u6e0e\u691f\u6a50\u724d\u8839\u7b03\u9ad1\u9ee9","duan":"\u7aef\u77ed\u953b\u6bb5\u65ad\u7f0e\u5f56\u6934\u7145\u7c16","dui":"\u5806\u5151\u961f\u5bf9\u603c\u619d\u7893","dun":"\u58a9\u5428\u8e72\u6566\u987f\u56e4\u949d\u76fe\u9041\u7096\u7818\u7905\u76f9\u9566\u8db8","duo":"\u6387\u54c6\u591a\u593a\u579b\u8eb2\u6735\u8dfa\u8235\u5241\u60f0\u5815\u5484\u54da\u7f0d\u67c1\u94ce\u88f0\u8e31","e":"\u86fe\u5ce8\u9e45\u4fc4\u989d\u8bb9\u5a25\u6076\u5384\u627c\u904f\u9102\u997f\u5669\u8c14\u57a9\u57ad\u82ca\u83aa\u843c\u5443\u6115\u5c59\u5a40\u8f6d\u66f7\u816d\u786a\u9507\u9537\u9e57\u989a\u9cc4","en":"\u6069\u84bd\u6441\u5514\u55ef","er":"\u800c\u513f\u8033\u5c14\u9975\u6d31\u4e8c\u8d30\u8fe9\u73e5\u94d2\u9e38\u9c95","fa":"\u53d1\u7f5a\u7b4f\u4f10\u4e4f\u9600\u6cd5\u73d0\u57a1\u781d","fan":"\u85e9\u5e06\u756a\u7ffb\u6a0a\u77fe\u9492\u7e41\u51e1\u70e6\u53cd\u8fd4\u8303\u8d29\u72af\u996d\u6cdb\u8629\u5e61\u72ad\u68b5\u6535\u71d4\u7548\u8e6f","fang":"\u574a\u82b3\u65b9\u80aa\u623f\u9632\u59a8\u4eff\u8bbf\u7eba\u653e\u531a\u90a1\u5f77\u94ab\u822b\u9c82","fei":"\u83f2\u975e\u5561\u98de\u80a5\u532a\u8bfd\u5420\u80ba\u5e9f\u6cb8\u8d39\u82be\u72d2\u60b1\u6ddd\u5983\u7ecb\u7eef\u69a7\u8153\u6590\u6249\u7953\u7829\u9544\u75f1\u871a\u7bda\u7fe1\u970f\u9cb1","fen":"\u82ac\u915a\u5429\u6c1b\u5206\u7eb7\u575f\u711a\u6c7e\u7c89\u594b\u4efd\u5fff\u6124\u7caa\u507e\u7035\u68fc\u610d\u9cbc\u9f22","feng":"\u4e30\u5c01\u67ab\u8702\u5cf0\u950b\u98ce\u75af\u70fd\u9022\u51af\u7f1d\u8bbd\u5949\u51e4\u4ff8\u9146\u8451\u6ca3\u781c","fu":"\u4f5b\u5426\u592b\u6577\u80a4\u5b75\u6276\u62c2\u8f90\u5e45\u6c1f\u7b26\u4f0f\u4fd8\u670d\u6d6e\u6daa\u798f\u88b1\u5f17\u752b\u629a\u8f85\u4fef\u91dc\u65a7\u812f\u8151\u5e9c\u8150\u8d74\u526f\u8986\u8d4b\u590d\u5085\u4ed8\u961c\u7236\u8179\u8d1f\u5bcc\u8ba3\u9644\u5987\u7f1a\u5490\u5310\u51eb\u90db\u8299\u82fb\u832f\u83a9\u83d4\u544b\u5e5e\u6ecf\u8274\u5b5a\u9a78\u7ec2\u6874\u8d59\u9efb\u9efc\u7f58\u7a03\u99a5\u864d\u86a8\u8709\u8760\u876e\u9eb8\u8dba\u8dd7\u9cc6","ga":"\u5676\u560e\u86e4\u5c2c\u5477\u5c15\u5c1c\u65ee\u9486","gai":"\u8be5\u6539\u6982\u9499\u76d6\u6e89\u4e10\u9654\u5793\u6224\u8d45\u80f2","gan":"\u5e72\u7518\u6746\u67d1\u7aff\u809d\u8d76\u611f\u79c6\u6562\u8d63\u5769\u82f7\u5c34\u64c0\u6cd4\u6de6\u6f89\u7ec0\u6a44\u65f0\u77f8\u75b3\u9150","gang":"\u5188\u521a\u94a2\u7f38\u809b\u7eb2\u5c97\u6e2f\u6206\u7f61\u9883\u7b7b","gong":"\u6760\u5de5\u653b\u529f\u606d\u9f9a\u4f9b\u8eac\u516c\u5bab\u5f13\u5de9\u6c5e\u62f1\u8d21\u5171\u857b\u5efe\u54a3\u73d9\u80b1\u86a3\u86e9\u89e5","gao":"\u7bd9\u768b\u9ad8\u818f\u7f94\u7cd5\u641e\u9550\u7a3f\u544a\u777e\u8bf0\u90dc\u84bf\u85c1\u7f1f\u69d4\u69c1\u6772\u9506","ge":"\u54e5\u6b4c\u6401\u6208\u9e3d\u80f3\u7599\u5272\u9769\u845b\u683c\u9601\u9694\u94ec\u4e2a\u5404\u9b32\u4ee1\u54ff\u5865\u55dd\u7ea5\u643f\u8188\u784c\u94ea\u9549\u88bc\u988c\u867c\u8238\u9abc\u9ac2","gei":"\u7ed9","gen":"\u6839\u8ddf\u4e98\u831b\u54cf\u826e","geng":"\u8015\u66f4\u5e9a\u7fb9\u57c2\u803f\u6897\u54fd\u8d53\u9ca0","gou":"\u94a9\u52fe\u6c9f\u82df\u72d7\u57a2\u6784\u8d2d\u591f\u4f5d\u8bdf\u5ca3\u9058\u5abe\u7f11\u89cf\u5f40\u9e32\u7b31\u7bdd\u97b2","gu":"\u8f9c\u83c7\u5495\u7b8d\u4f30\u6cbd\u5b64\u59d1\u9f13\u53e4\u86ca\u9aa8\u8c37\u80a1\u6545\u987e\u56fa\u96c7\u560f\u8bc2\u83f0\u54cc\u5d2e\u6c69\u688f\u8f71\u726f\u727f\u80cd\u81cc\u6bc2\u77bd\u7f5f\u94b4\u9522\u74e0\u9e2a\u9e44\u75fc\u86c4\u9164\u89da\u9cb4\u9ab0\u9e58","gua":"\u522e\u74dc\u5250\u5be1\u6302\u8902\u5366\u8bd6\u5471\u681d\u9e39","guai":"\u4e56\u62d0\u602a\u54d9","guan":"\u68fa\u5173\u5b98\u51a0\u89c2\u7ba1\u9986\u7f50\u60ef\u704c\u8d2f\u500c\u839e\u63bc\u6dab\u76e5\u9e73\u9ccf","guang":"\u5149\u5e7f\u901b\u72b7\u6844\u80f1\u7592","gui":"\u7470\u89c4\u572d\u7845\u5f52\u9f9f\u95fa\u8f68\u9b3c\u8be1\u7678\u6842\u67dc\u8dea\u8d35\u523d\u5326\u523f\u5e8b\u5b84\u59ab\u6867\u7085\u6677\u7688\u7c0b\u9c91\u9cdc","gun":"\u8f8a\u6eda\u68cd\u4e28\u886e\u7ef2\u78d9\u9ca7","guo":"\u9505\u90ed\u56fd\u679c\u88f9\u8fc7\u9998\u8803\u57da\u63b4\u5459\u56d7\u5e3c\u5d1e\u7313\u6901\u8662\u951e\u8052\u872e\u873e\u8748","ha":"\u54c8","hai":"\u9ab8\u5b69\u6d77\u6c26\u4ea5\u5bb3\u9a87\u54b4\u55e8\u988f\u91a2","han":"\u9163\u61a8\u90af\u97e9\u542b\u6db5\u5bd2\u51fd\u558a\u7f55\u7ff0\u64bc\u634d\u65f1\u61be\u608d\u710a\u6c57\u6c49\u9097\u83e1\u6496\u961a\u701a\u6657\u7113\u9894\u86b6\u9f3e","hen":"\u592f\u75d5\u5f88\u72e0\u6068","hang":"\u676d\u822a\u6c86\u7ed7\u73e9\u6841","hao":"\u58d5\u568e\u8c6a\u6beb\u90dd\u597d\u8017\u53f7\u6d69\u8585\u55e5\u5686\u6fe0\u704f\u660a\u7693\u98a2\u869d","he":"\u5475\u559d\u8377\u83cf\u6838\u79be\u548c\u4f55\u5408\u76d2\u8c89\u9602\u6cb3\u6db8\u8d6b\u8910\u9e64\u8d3a\u8bc3\u52be\u58d1\u85ff\u55d1\u55ec\u9616\u76cd\u86b5\u7fee","hei":"\u563f\u9ed1","heng":"\u54fc\u4ea8\u6a2a\u8861\u6052\u8a07\u8605","hong":"\u8f70\u54c4\u70d8\u8679\u9e3f\u6d2a\u5b8f\u5f18\u7ea2\u9ec9\u8ba7\u836d\u85a8\u95f3\u6cd3","hou":"\u5589\u4faf\u7334\u543c\u539a\u5019\u540e\u5820\u5f8c\u9005\u760a\u7bcc\u7cc7\u9c8e\u9aba","hu":"\u547c\u4e4e\u5ffd\u745a\u58f6\u846b\u80e1\u8774\u72d0\u7cca\u6e56\u5f27\u864e\u552c\u62a4\u4e92\u6caa\u6237\u51b1\u553f\u56eb\u5cb5\u7322\u6019\u60da\u6d52\u6ef9\u7425\u69f2\u8f77\u89f3\u70c0\u7173\u623d\u6248\u795c\u9e55\u9e71\u7b0f\u9190\u659b","hua":"\u82b1\u54d7\u534e\u733e\u6ed1\u753b\u5212\u5316\u8bdd\u5290\u6d4d\u9a85\u6866\u94e7\u7a1e","huai":"\u69d0\u5f8a\u6000\u6dee\u574f\u8fd8\u8e1d","huan":"\u6b22\u73af\u6853\u7f13\u6362\u60a3\u5524\u75ea\u8c62\u7115\u6da3\u5ba6\u5e7b\u90c7\u5942\u57b8\u64d0\u571c\u6d39\u6d63\u6f36\u5bf0\u902d\u7f33\u953e\u9ca9\u9b1f","huang":"\u8352\u614c\u9ec4\u78fa\u8757\u7c27\u7687\u51f0\u60f6\u714c\u6643\u5e4c\u604d\u8c0e\u968d\u5fa8\u6e5f\u6f62\u9051\u749c\u8093\u7640\u87e5\u7bc1\u9cc7","hui":"\u7070\u6325\u8f89\u5fbd\u6062\u86d4\u56de\u6bc1\u6094\u6167\u5349\u60e0\u6666\u8d3f\u79fd\u4f1a\u70e9\u6c47\u8bb3\u8bf2\u7ed8\u8bd9\u8334\u835f\u8559\u54d5\u5599\u96b3\u6d04\u5f57\u7f0b\u73f2\u6656\u605a\u867a\u87ea\u9ebe","hun":"\u8364\u660f\u5a5a\u9b42\u6d51\u6df7\u8be8\u9984\u960d\u6eb7\u7f17","huo":"\u8c41\u6d3b\u4f19\u706b\u83b7\u6216\u60d1\u970d\u8d27\u7978\u6509\u56af\u5925\u94ac\u952a\u956c\u8020\u8816","ji":"\u51fb\u573e\u57fa\u673a\u7578\u7a3d\u79ef\u7b95\u808c\u9965\u8ff9\u6fc0\u8ba5\u9e21\u59ec\u7ee9\u7f09\u5409\u6781\u68d8\u8f91\u7c4d\u96c6\u53ca\u6025\u75be\u6c72\u5373\u5ac9\u7ea7\u6324\u51e0\u810a\u5df1\u84df\u6280\u5180\u5b63\u4f0e\u796d\u5242\u60b8\u6d4e\u5bc4\u5bc2\u8ba1\u8bb0\u65e2\u5fcc\u9645\u5993\u7ee7\u7eaa\u5c45\u4e0c\u4e69\u525e\u4f76\u4f74\u8114\u58bc\u82a8\u82b0\u8401\u84ba\u857a\u638e\u53fd\u54ad\u54dc\u5527\u5c8c\u5d74\u6d0e\u5f50\u5c50\u9aa5\u757f\u7391\u696b\u6b9b\u621f\u6222\u8d4d\u89ca\u7284\u9f51\u77f6\u7f81\u5d47\u7a37\u7620\u7635\u866e\u7b08\u7b04\u66a8\u8dfb\u8dfd\u9701\u9c9a\u9cab\u9afb\u9e82","jia":"\u5609\u67b7\u5939\u4f73\u5bb6\u52a0\u835a\u988a\u8d3e\u7532\u94be\u5047\u7a3c\u4ef7\u67b6\u9a7e\u5ac1\u4f3d\u90cf\u62ee\u5cac\u6d43\u8fe6\u73c8\u621b\u80db\u605d\u94d7\u9553\u75c2\u86f1\u7b33\u8888\u8dcf","jian":"\u6b7c\u76d1\u575a\u5c16\u7b3a\u95f4\u714e\u517c\u80a9\u8270\u5978\u7f04\u8327\u68c0\u67ec\u78b1\u7877\u62e3\u6361\u7b80\u4fed\u526a\u51cf\u8350\u69db\u9274\u8df5\u8d31\u89c1\u952e\u7bad\u4ef6\u5065\u8230\u5251\u996f\u6e10\u6e85\u6da7\u5efa\u50ed\u8c0f\u8c2b\u83c5\u84b9\u641b\u56dd\u6e54\u8e47\u8b07\u7f23\u67a7\u67d9\u6957\u620b\u622c\u726e\u728d\u6bfd\u8171\u7751\u950f\u9e63\u88e5\u7b15\u7bb4\u7fe6\u8dbc\u8e3a\u9ca3\u97af","jiang":"\u50f5\u59dc\u5c06\u6d46\u6c5f\u7586\u848b\u6868\u5956\u8bb2\u5320\u9171\u964d\u8333\u6d1a\u7edb\u7f30\u729f\u7913\u8029\u7ce8\u8c47","jiao":"\u8549\u6912\u7901\u7126\u80f6\u4ea4\u90ca\u6d47\u9a84\u5a07\u56bc\u6405\u94f0\u77eb\u4fa5\u811a\u72e1\u89d2\u997a\u7f34\u7ede\u527f\u6559\u9175\u8f7f\u8f83\u53eb\u4f7c\u50ec\u832d\u6322\u564d\u5ce4\u5fbc\u59e3\u7e9f\u656b\u768e\u9e6a\u86df\u91ae\u8de4\u9c9b","jie":"\u7a96\u63ed\u63a5\u7686\u79f8\u8857\u9636\u622a\u52ab\u8282\u6854\u6770\u6377\u776b\u7aed\u6d01\u7ed3\u89e3\u59d0\u6212\u85c9\u82a5\u754c\u501f\u4ecb\u75a5\u8beb\u5c4a\u5048\u8ba6\u8bd8\u5588\u55df\u736c\u5a55\u5b51\u6840\u7352\u78a3\u9534\u7596\u88b7\u9889\u86a7\u7faf\u9c92\u9ab1\u9aeb","jin":"\u5dfe\u7b4b\u65a4\u91d1\u4eca\u6d25\u895f\u7d27\u9526\u4ec5\u8c28\u8fdb\u9773\u664b\u7981\u8fd1\u70ec\u6d78\u5c3d\u537a\u8369\u5807\u5664\u9991\u5ed1\u5997\u7f19\u747e\u69ff\u8d46\u89d0\u9485\u9513\u887f\u77dc","jing":"\u52b2\u8346\u5162\u830e\u775b\u6676\u9cb8\u4eac\u60ca\u7cbe\u7cb3\u7ecf\u4e95\u8b66\u666f\u9888\u9759\u5883\u656c\u955c\u5f84\u75c9\u9756\u7adf\u7ade\u51c0\u522d\u5106\u9631\u83c1\u734d\u61ac\u6cfe\u8ff3\u5f2a\u5a67\u80bc\u80eb\u8148\u65cc","jiong":"\u70af\u7a98\u5182\u8fe5\u6243","jiu":"\u63ea\u7a76\u7ea0\u7396\u97ed\u4e45\u7078\u4e5d\u9152\u53a9\u6551\u65e7\u81fc\u8205\u548e\u5c31\u759a\u50e6\u557e\u9604\u67e9\u6855\u9e6b\u8d73\u9b0f","ju":"\u97a0\u62d8\u72d9\u75bd\u9a79\u83ca\u5c40\u5480\u77e9\u4e3e\u6cae\u805a\u62d2\u636e\u5de8\u5177\u8ddd\u8e1e\u952f\u4ff1\u53e5\u60e7\u70ac\u5267\u5028\u8bb5\u82e3\u82f4\u8392\u63ac\u907d\u5c66\u741a\u67b8\u6910\u6998\u6989\u6a58\u728b\u98d3\u949c\u9514\u7aad\u88fe\u8d84\u91b5\u8e3d\u9f83\u96ce\u97ab","juan":"\u6350\u9e43\u5a1f\u5026\u7737\u5377\u7ee2\u9104\u72f7\u6d93\u684a\u8832\u9529\u954c\u96bd","jue":"\u6485\u652b\u6289\u6398\u5014\u7235\u89c9\u51b3\u8bc0\u7edd\u53a5\u5282\u8c32\u77cd\u8568\u5658\u5d1b\u7357\u5b53\u73cf\u6877\u6a5b\u721d\u9562\u8e76\u89d6","jun":"\u5747\u83cc\u94a7\u519b\u541b\u5cfb\u4fca\u7ae3\u6d5a\u90e1\u9a8f\u6343\u72fb\u76b2\u7b60\u9e87","ka":"\u5580\u5496\u5361\u4f67\u5494\u80e9","ke":"\u54af\u5777\u82db\u67ef\u68f5\u78d5\u9897\u79d1\u58f3\u54b3\u53ef\u6e34\u514b\u523b\u5ba2\u8bfe\u5ca2\u606a\u6e98\u9a92\u7f02\u73c2\u8f72\u6c2a\u778c\u94b6\u75b4\u7aa0\u874c\u9ac1","kai":"\u5f00\u63e9\u6977\u51ef\u6168\u5240\u57b2\u8488\u5ffe\u607a\u94e0\u950e","kan":"\u520a\u582a\u52d8\u574e\u780d\u770b\u4f83\u51f5\u83b0\u83b6\u6221\u9f9b\u77b0","kang":"\u5eb7\u6177\u7ce0\u625b\u6297\u4ea2\u7095\u5751\u4f09\u95f6\u94aa","kao":"\u8003\u62f7\u70e4\u9760\u5c3b\u6832\u7292\u94d0","ken":"\u80af\u5543\u57a6\u6073\u57a0\u88c9\u9880","keng":"\u542d\u5fd0\u94ff","kong":"\u7a7a\u6050\u5b54\u63a7\u5025\u5d06\u7b9c","kou":"\u62a0\u53e3\u6263\u5bc7\u82a4\u853b\u53e9\u770d\u7b58","ku":"\u67af\u54ed\u7a9f\u82e6\u9177\u5e93\u88e4\u5233\u5800\u55be\u7ed4\u9ab7","kua":"\u5938\u57ae\u630e\u8de8\u80ef\u4f89","kuai":"\u5757\u7b77\u4fa9\u5feb\u84af\u90d0\u8489\u72ef\u810d","kuan":"\u5bbd\u6b3e\u9acb","kuang":"\u5321\u7b50\u72c2\u6846\u77ff\u7736\u65f7\u51b5\u8bd3\u8bf3\u909d\u5739\u593c\u54d0\u7ea9\u8d36","kui":"\u4e8f\u76d4\u5cbf\u7aa5\u8475\u594e\u9b41\u5080\u9988\u6127\u6e83\u9997\u532e\u5914\u9697\u63c6\u55b9\u559f\u609d\u6126\u9615\u9035\u668c\u777d\u8069\u8770\u7bd1\u81fe\u8dec","kun":"\u5764\u6606\u6346\u56f0\u6083\u9603\u7428\u951f\u918c\u9cb2\u9ae1","kuo":"\u62ec\u6269\u5ed3\u9614\u86de","la":"\u5783\u62c9\u5587\u8721\u814a\u8fa3\u5566\u524c\u647a\u908b\u65ef\u782c\u760c","lai":"\u83b1\u6765\u8d56\u5d03\u5f95\u6d9e\u6fd1\u8d49\u7750\u94fc\u765e\u7c41","lan":"\u84dd\u5a6a\u680f\u62e6\u7bee\u9611\u5170\u6f9c\u8c30\u63fd\u89c8\u61d2\u7f06\u70c2\u6ee5\u5549\u5c9a\u61d4\u6f24\u6984\u6593\u7f71\u9567\u8934","lang":"\u7405\u6994\u72fc\u5eca\u90ce\u6717\u6d6a\u83a8\u8497\u5577\u9606\u9512\u7a02\u8782","lao":"\u635e\u52b3\u7262\u8001\u4f6c\u59e5\u916a\u70d9\u6d9d\u5520\u5d02\u6833\u94d1\u94f9\u75e8\u91aa","le":"\u52d2\u4e50\u808b\u4ec2\u53fb\u561e\u6cd0\u9cd3","lei":"\u96f7\u956d\u857e\u78ca\u7d2f\u5121\u5792\u64c2\u7c7b\u6cea\u7fb8\u8bd4\u837d\u54a7\u6f2f\u5ad8\u7f27\u6a91\u8012\u9179","ling":"\u68f1\u51b7\u62ce\u73b2\u83f1\u96f6\u9f84\u94c3\u4f36\u7f9a\u51cc\u7075\u9675\u5cad\u9886\u53e6\u4ee4\u9143\u5844\u82d3\u5464\u56f9\u6ce0\u7eeb\u67c3\u68c2\u74f4\u8046\u86c9\u7fce\u9cae","leng":"\u695e\u6123","li":"\u5398\u68a8\u7281\u9ece\u7bf1\u72f8\u79bb\u6f13\u7406\u674e\u91cc\u9ca4\u793c\u8389\u8354\u540f\u6817\u4e3d\u5389\u52b1\u783e\u5386\u5229\u5088\u4f8b\u4fd0\u75e2\u7acb\u7c92\u6ca5\u96b6\u529b\u7483\u54e9\u4fea\u4fda\u90e6\u575c\u82c8\u8385\u84e0\u85dc\u6369\u5456\u5533\u55b1\u7301\u6ea7\u6fa7\u9026\u5a0c\u5ae0\u9a8a\u7f21\u73de\u67a5\u680e\u8f79\u623e\u783a\u8a48\u7f79\u9502\u9e42\u75a0\u75ac\u86ce\u870a\u8821\u7b20\u7be5\u7c9d\u91b4\u8dde\u96f3\u9ca1\u9ce2\u9ee7","lian":"\u4fe9\u8054\u83b2\u8fde\u9570\u5ec9\u601c\u6d9f\u5e18\u655b\u8138\u94fe\u604b\u70bc\u7ec3\u631b\u8539\u5941\u6f4b\u6fc2\u5a08\u740f\u695d\u6b93\u81c1\u81a6\u88e2\u880a\u9ca2","liang":"\u7cae\u51c9\u6881\u7cb1\u826f\u4e24\u8f86\u91cf\u667e\u4eae\u8c05\u589a\u690b\u8e09\u9753\u9b49","liao":"\u64a9\u804a\u50da\u7597\u71ce\u5be5\u8fbd\u6f66\u4e86\u6482\u9563\u5ed6\u6599\u84fc\u5c25\u5639\u7360\u5bee\u7f2d\u948c\u9e69\u8022","lie":"\u5217\u88c2\u70c8\u52a3\u730e\u51bd\u57d2\u6d0c\u8d94\u8e90\u9b23","lin":"\u7433\u6797\u78f7\u9716\u4e34\u90bb\u9cde\u6dcb\u51db\u8d41\u541d\u853a\u5d99\u5eea\u9074\u6aa9\u8f9a\u77b5\u7cbc\u8e8f\u9e9f","liu":"\u6e9c\u7409\u69b4\u786b\u998f\u7559\u5218\u7624\u6d41\u67f3\u516d\u62a1\u507b\u848c\u6cd6\u6d4f\u905b\u9a9d\u7efa\u65d2\u7198\u950d\u954f\u9e68\u938f","long":"\u9f99\u804b\u5499\u7b3c\u7abf\u9686\u5784\u62e2\u9647\u5f04\u5785\u830f\u6cf7\u73d1\u680a\u80e7\u783b\u7643","lou":"\u697c\u5a04\u6402\u7bd3\u6f0f\u964b\u55bd\u5d5d\u9542\u7618\u8027\u877c\u9ac5","lu":"\u82a6\u5362\u9885\u5e90\u7089\u63b3\u5364\u864f\u9c81\u9e93\u788c\u9732\u8def\u8d42\u9e7f\u6f5e\u7984\u5f55\u9646\u622e\u5786\u6445\u64b8\u565c\u6cf8\u6e0c\u6f09\u7490\u680c\u6a79\u8f73\u8f82\u8f98\u6c07\u80ea\u9565\u9e2c\u9e6d\u7c0f\u823b\u9c88","lv":"\u9a74\u5415\u94dd\u4fa3\u65c5\u5c65\u5c61\u7f15\u8651\u6c2f\u5f8b\u7387\u6ee4\u7eff\u634b\u95fe\u6988\u8182\u7a06\u891b","luan":"\u5ce6\u5b6a\u6ee6\u5375\u4e71\u683e\u9e3e\u92ae","lue":"\u63a0\u7565\u950a","lun":"\u8f6e\u4f26\u4ed1\u6ca6\u7eb6\u8bba\u56f5","luo":"\u841d\u87ba\u7f57\u903b\u9523\u7ba9\u9aa1\u88f8\u843d\u6d1b\u9a86\u7edc\u502e\u8366\u645e\u7321\u6cfa\u6924\u8136\u9559\u7630\u96d2","ma":"\u5988\u9ebb\u739b\u7801\u8682\u9a6c\u9a82\u561b\u5417\u551b\u72b8\u5b37\u6769\u9ebd","mai":"\u57cb\u4e70\u9ea6\u5356\u8fc8\u8109\u52a2\u836c\u54aa\u973e","man":"\u7792\u9992\u86ee\u6ee1\u8513\u66fc\u6162\u6f2b\u8c29\u5881\u5e54\u7f26\u71b3\u9558\u989f\u87a8\u9cd7\u9794","mang":"\u8292\u832b\u76f2\u5fd9\u83bd\u9099\u6f2d\u6726\u786d\u87d2","meng":"\u6c13\u840c\u8499\u6aac\u76df\u9530\u731b\u68a6\u5b5f\u52d0\u750d\u77a2\u61f5\u791e\u867b\u8722\u8813\u824b\u8268\u9efe","miao":"\u732b\u82d7\u63cf\u7784\u85d0\u79d2\u6e3a\u5e99\u5999\u55b5\u9088\u7f08\u7f2a\u676a\u6dfc\u7707\u9e4b\u8731","mao":"\u8305\u951a\u6bdb\u77db\u94c6\u536f\u8302\u5192\u5e3d\u8c8c\u8d38\u4f94\u88a4\u52d6\u8306\u5cc1\u7441\u6634\u7266\u8004\u65c4\u61cb\u7780\u86d1\u8765\u87ca\u9ae6","me":"\u4e48","mei":"\u73ab\u679a\u6885\u9176\u9709\u7164\u6ca1\u7709\u5a92\u9541\u6bcf\u7f8e\u6627\u5bd0\u59b9\u5a9a\u5776\u8393\u5d4b\u7338\u6d7c\u6e44\u6963\u9545\u9e5b\u8882\u9b45","men":"\u95e8\u95f7\u4eec\u626a\u739f\u7116\u61d1\u9494","mi":"\u772f\u919a\u9761\u7cdc\u8ff7\u8c1c\u5f25\u7c73\u79d8\u89c5\u6ccc\u871c\u5bc6\u5e42\u8288\u5196\u8c27\u863c\u5627\u7315\u736f\u6c68\u5b93\u5f2d\u8112\u6549\u7cf8\u7e3b\u9e8b","mian":"\u68c9\u7720\u7ef5\u5195\u514d\u52c9\u5a29\u7f05\u9762\u6c94\u6e4e\u817c\u7704","mie":"\u8511\u706d\u54a9\u881b\u7bfe","min":"\u6c11\u62bf\u76bf\u654f\u60af\u95fd\u82e0\u5cb7\u95f5\u6cef\u73c9","ming":"\u660e\u879f\u9e23\u94ed\u540d\u547d\u51a5\u8317\u6e9f\u669d\u7791\u9169","miu":"\u8c2c","mo":"\u6478\u6479\u8611\u6a21\u819c\u78e8\u6469\u9b54\u62b9\u672b\u83ab\u58a8\u9ed8\u6cab\u6f20\u5bde\u964c\u8c1f\u8309\u84e6\u998d\u5aeb\u9546\u79e3\u763c\u8031\u87c6\u8c8a\u8c98","mou":"\u8c0b\u725f\u67d0\u53b6\u54de\u5a7a\u7738\u936a","mu":"\u62c7\u7261\u4ea9\u59c6\u6bcd\u5893\u66ae\u5e55\u52df\u6155\u6728\u76ee\u7766\u7267\u7a46\u4eeb\u82dc\u5452\u6c90\u6bea\u94bc","na":"\u62ff\u54ea\u5450\u94a0\u90a3\u5a1c\u7eb3\u5185\u637a\u80ad\u954e\u8872\u7bac","nai":"\u6c16\u4e43\u5976\u8010\u5948\u9f10\u827f\u8418\u67f0","nan":"\u5357\u7537\u96be\u56ca\u5583\u56e1\u6960\u8169\u877b\u8d67","nao":"\u6320\u8111\u607c\u95f9\u5b6c\u57b4\u7331\u7459\u7847\u94d9\u86f2","ne":"\u6dd6\u5462\u8bb7","nei":"\u9981","nen":"\u5ae9\u80fd\u6798\u6041","ni":"\u59ae\u9713\u502a\u6ce5\u5c3c\u62df\u4f60\u533f\u817b\u9006\u6eba\u4f32\u576d\u730a\u6029\u6ee0\u6635\u65ce\u7962\u615d\u7768\u94cc\u9cb5","nian":"\u852b\u62c8\u5e74\u78be\u64b5\u637b\u5ff5\u5eff\u8f87\u9ecf\u9c87\u9cb6","niang":"\u5a18\u917f","niao":"\u9e1f\u5c3f\u8311\u5b32\u8132\u8885","nie":"\u634f\u8042\u5b7d\u556e\u954a\u954d\u6d85\u4e5c\u9667\u8616\u55eb\u8080\u989e\u81ec\u8e51","nin":"\u60a8\u67e0","ning":"\u72de\u51dd\u5b81\u62e7\u6cde\u4f5e\u84e5\u549b\u752f\u804d","niu":"\u725b\u626d\u94ae\u7ebd\u72c3\u5ff8\u599e\u86b4","nong":"\u8113\u6d53\u519c\u4fac","nu":"\u5974\u52aa\u6012\u5476\u5e11\u5f29\u80ec\u5b65\u9a7d","nv":"\u5973\u6067\u9495\u8844","nuan":"\u6696","nuenue":"\u8650","nue":"\u759f\u8c11","nuo":"\u632a\u61e6\u7cef\u8bfa\u50a9\u6426\u558f\u9518","ou":"\u54e6\u6b27\u9e25\u6bb4\u85d5\u5455\u5076\u6ca4\u6004\u74ef\u8026","pa":"\u556a\u8db4\u722c\u5e15\u6015\u7436\u8469\u7b62","pai":"\u62cd\u6392\u724c\u5f98\u6e43\u6d3e\u4ff3\u848e","pan":"\u6500\u6f58\u76d8\u78d0\u76fc\u7554\u5224\u53db\u723f\u6cee\u88a2\u897b\u87e0\u8e52","pang":"\u4e53\u5e9e\u65c1\u802a\u80d6\u6ec2\u9004","pao":"\u629b\u5486\u5228\u70ae\u888d\u8dd1\u6ce1\u530f\u72cd\u5e96\u812c\u75b1","pei":"\u5478\u80da\u57f9\u88f4\u8d54\u966a\u914d\u4f69\u6c9b\u638a\u8f94\u5e14\u6de0\u65c6\u952b\u9185\u9708","pen":"\u55b7\u76c6\u6e53","peng":"\u7830\u62a8\u70f9\u6f8e\u5f6d\u84ec\u68da\u787c\u7bf7\u81a8\u670b\u9e4f\u6367\u78b0\u576f\u580b\u562d\u6026\u87db","pi":"\u7812\u9739\u6279\u62ab\u5288\u7435\u6bd7\u5564\u813e\u75b2\u76ae\u5339\u75de\u50fb\u5c41\u8b6c\u4e15\u9674\u90b3\u90eb\u572e\u9f19\u64d7\u567c\u5e80\u5ab2\u7eb0\u6787\u7513\u7765\u7f74\u94cd\u75e6\u7656\u758b\u868d\u8c94","pian":"\u7bc7\u504f\u7247\u9a97\u8c1d\u9a88\u728f\u80fc\u890a\u7fe9\u8e41","piao":"\u98d8\u6f02\u74e2\u7968\u527d\u560c\u5ad6\u7f25\u6b8d\u779f\u87b5","pie":"\u6487\u77a5\u4e3f\u82e4\u6c15","pin":"\u62fc\u9891\u8d2b\u54c1\u8058\u62da\u59d8\u5ad4\u6980\u725d\u98a6","ping":"\u4e52\u576a\u82f9\u840d\u5e73\u51ed\u74f6\u8bc4\u5c4f\u4fdc\u5a09\u67b0\u9c86","po":"\u5761\u6cfc\u9887\u5a46\u7834\u9b44\u8feb\u7c95\u53f5\u9131\u6ea5\u73c0\u948b\u94b7\u76a4\u7b38","pou":"\u5256\u88d2\u8e23","pu":"\u6251\u94fa\u4ec6\u8386\u8461\u83e9\u84b2\u57d4\u6734\u5703\u666e\u6d66\u8c31\u66dd\u7011\u530d\u5657\u6fee\u749e\u6c06\u9564\u9568\u8e7c","qi":"\u671f\u6b3a\u6816\u621a\u59bb\u4e03\u51c4\u6f06\u67d2\u6c8f\u5176\u68cb\u5947\u6b67\u7566\u5d0e\u8110\u9f50\u65d7\u7948\u7941\u9a91\u8d77\u5c82\u4e5e\u4f01\u542f\u5951\u780c\u5668\u6c14\u8fc4\u5f03\u6c7d\u6ce3\u8bab\u4e9f\u4e93\u573b\u8291\u840b\u847a\u5601\u5c7a\u5c90\u6c54\u6dc7\u9a90\u7eee\u742a\u7426\u675e\u6864\u69ed\u6b39\u797a\u61a9\u789b\u86f4\u871e\u7da6\u7dae\u8dbf\u8e4a\u9ccd\u9e92","qia":"\u6390\u6070\u6d3d\u845c","qian":"\u7275\u6266\u948e\u94c5\u5343\u8fc1\u7b7e\u4edf\u8c26\u4e7e\u9ed4\u94b1\u94b3\u524d\u6f5c\u9063\u6d45\u8c34\u5811\u5d4c\u6b20\u6b49\u4f65\u9621\u828a\u82a1\u8368\u63ae\u5c8d\u60ad\u614a\u9a9e\u6434\u8930\u7f31\u6920\u80b7\u6106\u94a4\u8654\u7b9d","qiang":"\u67aa\u545b\u8154\u7f8c\u5899\u8537\u5f3a\u62a2\u5af1\u6a2f\u6217\u709d\u9516\u9535\u956a\u8941\u8723\u7f9f\u8deb\u8dc4","qiao":"\u6a47\u9539\u6572\u6084\u6865\u77a7\u4e54\u4fa8\u5de7\u9798\u64ac\u7fd8\u5ced\u4fcf\u7a8d\u5281\u8bee\u8c2f\u835e\u6100\u6194\u7f32\u6a35\u6bf3\u7857\u8df7\u9792","qie":"\u5207\u8304\u4e14\u602f\u7a83\u90c4\u553c\u60ec\u59be\u6308\u9532\u7ba7","qin":"\u94a6\u4fb5\u4eb2\u79e6\u7434\u52e4\u82b9\u64d2\u79bd\u5bdd\u6c81\u82a9\u84c1\u8572\u63ff\u5423\u55ea\u5659\u6eb1\u6a8e\u8793\u887e","qing":"\u9752\u8f7b\u6c22\u503e\u537f\u6e05\u64ce\u6674\u6c30\u60c5\u9877\u8bf7\u5e86\u5029\u82d8\u570a\u6aa0\u78ec\u873b\u7f44\u7b90\u8b26\u9cad\u9ee5","qiong":"\u743c\u7a77\u909b\u8315\u7a79\u7b47\u928e","qiu":"\u79cb\u4e18\u90b1\u7403\u6c42\u56da\u914b\u6cc5\u4fc5\u6c3d\u5def\u827d\u72b0\u6e6b\u9011\u9052\u6978\u8d47\u9e20\u866c\u86af\u8764\u88d8\u7cd7\u9cc5\u9f3d","qu":"\u8d8b\u533a\u86c6\u66f2\u8eaf\u5c48\u9a71\u6e20\u53d6\u5a36\u9f8b\u8da3\u53bb\u8bce\u52ac\u8556\u8627\u5c96\u8862\u9612\u74a9\u89d1\u6c0d\u795b\u78f2\u766f\u86d0\u883c\u9eb4\u77bf\u9ee2","quan":"\u5708\u98a7\u6743\u919b\u6cc9\u5168\u75ca\u62f3\u72ac\u5238\u529d\u8be0\u8343\u737e\u609b\u7efb\u8f81\u754e\u94e8\u8737\u7b4c\u9b08","que":"\u7f3a\u7094\u7638\u5374\u9e4a\u69b7\u786e\u96c0\u9619\u60ab","qun":"\u88d9\u7fa4\u9021","ran":"\u7136\u71c3\u5189\u67d3\u82d2\u9aef","rang":"\u74e4\u58e4\u6518\u56b7\u8ba9\u79b3\u7a70","rao":"\u9976\u6270\u7ed5\u835b\u5a06\u6861","ruo":"\u60f9\u82e5\u5f31","re":"\u70ed\u504c","ren":"\u58ec\u4ec1\u4eba\u5fcd\u97e7\u4efb\u8ba4\u5203\u598a\u7eab\u4ede\u834f\u845a\u996a\u8f6b\u7a14\u887d","reng":"\u6254\u4ecd","ri":"\u65e5","rong":"\u620e\u8338\u84c9\u8363\u878d\u7194\u6eb6\u5bb9\u7ed2\u5197\u5d58\u72e8\u7f1b\u6995\u877e","rou":"\u63c9\u67d4\u8089\u7cc5\u8e42\u97a3","ru":"\u8339\u8815\u5112\u5b7a\u5982\u8fb1\u4e73\u6c5d\u5165\u8925\u84d0\u85b7\u5685\u6d33\u6ebd\u6fe1\u94f7\u8966\u98a5","ruan":"\u8f6f\u962e\u670a","rui":"\u854a\u745e\u9510\u82ae\u8564\u777f\u868b","run":"\u95f0\u6da6","sa":"\u6492\u6d12\u8428\u5345\u4ee8\u6332\u98d2","sai":"\u816e\u9cc3\u585e\u8d5b\u567b","san":"\u4e09\u53c1\u4f1e\u6563\u5f61\u9993\u6c35\u6bf5\u7cc1\u9730","sang":"\u6851\u55d3\u4e27\u6421\u78c9\u98a1","sao":"\u6414\u9a9a\u626b\u5ac2\u57fd\u81ca\u7619\u9ccb","se":"\u745f\u8272\u6da9\u556c\u94e9\u94ef\u7a51","sen":"\u68ee","seng":"\u50e7","sha":"\u838e\u7802\u6740\u5239\u6c99\u7eb1\u50bb\u5565\u715e\u810e\u6b43\u75e7\u88df\u970e\u9ca8","shai":"\u7b5b\u6652\u917e","shan":"\u73ca\u82eb\u6749\u5c71\u5220\u717d\u886b\u95ea\u9655\u64c5\u8d61\u81b3\u5584\u6c55\u6247\u7f2e\u5261\u8baa\u912f\u57cf\u829f\u6f78\u59d7\u9a9f\u81bb\u9490\u759d\u87ee\u8222\u8dda\u9cdd","shang":"\u5892\u4f24\u5546\u8d4f\u664c\u4e0a\u5c1a\u88f3\u57a7\u7ef1\u6b87\u71b5\u89de","shao":"\u68a2\u634e\u7a0d\u70e7\u828d\u52fa\u97f6\u5c11\u54e8\u90b5\u7ecd\u52ad\u82d5\u6f72\u86f8\u7b24\u7b72\u8244","she":"\u5962\u8d4a\u86c7\u820c\u820d\u8d66\u6444\u5c04\u6151\u6d89\u793e\u8bbe\u538d\u4f58\u731e\u7572\u9e9d","shen":"\u7837\u7533\u547b\u4f38\u8eab\u6df1\u5a20\u7ec5\u795e\u6c88\u5ba1\u5a76\u751a\u80be\u614e\u6e17\u8bdc\u8c02\u5432\u54c2\u6e16\u6939\u77e7\u8703","sheng":"\u58f0\u751f\u7525\u7272\u5347\u7ef3\u7701\u76db\u5269\u80dc\u5723\u4e1e\u6e11\u5ab5\u771a\u7b19","shi":"\u5e08\u5931\u72ee\u65bd\u6e7f\u8bd7\u5c38\u8671\u5341\u77f3\u62fe\u65f6\u4ec0\u98df\u8680\u5b9e\u8bc6\u53f2\u77e2\u4f7f\u5c4e\u9a76\u59cb\u5f0f\u793a\u58eb\u4e16\u67ff\u4e8b\u62ed\u8a93\u901d\u52bf\u662f\u55dc\u566c\u9002\u4ed5\u4f8d\u91ca\u9970\u6c0f\u5e02\u6043\u5ba4\u89c6\u8bd5\u8c25\u57d8\u83b3\u84cd\u5f11\u5511\u9963\u8f7c\u8006\u8d33\u70bb\u793b\u94c8\u94ca\u87ab\u8210\u7b6e\u8c55\u9ca5\u9cba","shou":"\u6536\u624b\u9996\u5b88\u5bff\u6388\u552e\u53d7\u7626\u517d\u624c\u72e9\u7ef6\u824f","shu":"\u852c\u67a2\u68b3\u6b8a\u6292\u8f93\u53d4\u8212\u6dd1\u758f\u4e66\u8d4e\u5b70\u719f\u85af\u6691\u66d9\u7f72\u8700\u9ecd\u9f20\u5c5e\u672f\u8ff0\u6811\u675f\u620d\u7ad6\u5885\u5eb6\u6570\u6f31\u6055\u500f\u587e\u83fd\u5fc4\u6cad\u6d91\u6f8d\u59dd\u7ebe\u6bf9\u8167\u6bb3\u956f\u79eb\u9e6c","shua":"\u5237\u800d\u5530\u6dae","shuai":"\u6454\u8870\u7529\u5e05\u87c0","shuan":"\u6813\u62f4\u95e9","shuang":"\u971c\u53cc\u723d\u5b40","shui":"\u8c01\u6c34\u7761\u7a0e","shun":"\u542e\u77ac\u987a\u821c\u6042","shuo":"\u8bf4\u7855\u6714\u70c1\u84b4\u6420\u55cd\u6fef\u5981\u69ca\u94c4","si":"\u65af\u6495\u5636\u601d\u79c1\u53f8\u4e1d\u6b7b\u8086\u5bfa\u55e3\u56db\u4f3a\u4f3c\u9972\u5df3\u53ae\u4fdf\u5155\u83e5\u549d\u6c5c\u6cd7\u6f8c\u59d2\u9a77\u7f0c\u7940\u7960\u9536\u9e36\u801c\u86f3\u7b25","song":"\u677e\u8038\u6002\u9882\u9001\u5b8b\u8bbc\u8bf5\u51c7\u83d8\u5d27\u5d69\u5fea\u609a\u6dde\u7ae6","sou":"\u641c\u8258\u64de\u55fd\u53df\u55d6\u55fe\u998a\u6eb2\u98d5\u778d\u953c\u878b","su":"\u82cf\u9165\u4fd7\u7d20\u901f\u7c9f\u50f3\u5851\u6eaf\u5bbf\u8bc9\u8083\u5919\u8c21\u850c\u55c9\u612b\u7c0c\u89eb\u7a23","suan":"\u9178\u849c\u7b97","sui":"\u867d\u968b\u968f\u7ee5\u9ad3\u788e\u5c81\u7a57\u9042\u96a7\u795f\u84d1\u51ab\u8c07\u6fc9\u9083\u71e7\u772d\u7762","sun":"\u5b59\u635f\u7b0b\u836a\u72f2\u98e7\u69ab\u8de3\u96bc","suo":"\u68ad\u5506\u7f29\u7410\u7d22\u9501\u6240\u5522\u55e6\u5a11\u686b\u7743\u7fa7","ta":"\u584c\u4ed6\u5b83\u5979\u5854\u736d\u631e\u8e4b\u8e0f\u95fc\u6ebb\u9062\u69bb\u6c93","tai":"\u80ce\u82d4\u62ac\u53f0\u6cf0\u915e\u592a\u6001\u6c70\u90b0\u85b9\u80bd\u70b1\u949b\u8dc6\u9c90","tan":"\u574d\u644a\u8d2a\u762b\u6ee9\u575b\u6a80\u75f0\u6f6d\u8c2d\u8c08\u5766\u6bef\u8892\u78b3\u63a2\u53f9\u70ad\u90ef\u8548\u6619\u94bd\u952c\u8983","tang":"\u6c64\u5858\u642a\u5802\u68e0\u819b\u5510\u7cd6\u50a5\u9967\u6e8f\u746d\u94f4\u9557\u8025\u8797\u87b3\u7fb0\u91a3","thang":"\u5018\u8eba\u6dcc","theng":"\u8d9f\u70eb","tao":"\u638f\u6d9b\u6ed4\u7ee6\u8404\u6843\u9003\u6dd8\u9676\u8ba8\u5957\u6311\u9f17\u5555\u97ec\u9955","te":"\u7279","teng":"\u85e4\u817e\u75bc\u8a8a\u6ed5","ti":"\u68af\u5254\u8e22\u9511\u63d0\u9898\u8e44\u557c\u4f53\u66ff\u568f\u60d5\u6d95\u5243\u5c49\u8351\u608c\u9016\u7ee8\u7f07\u9e48\u88fc\u918d","tian":"\u5929\u6dfb\u586b\u7530\u751c\u606c\u8214\u8146\u63ad\u5fdd\u9617\u6b84\u754b\u94bf\u86ba","tiao":"\u6761\u8fe2\u773a\u8df3\u4f7b\u7967\u94eb\u7a95\u9f86\u9ca6","tie":"\u8d34\u94c1\u5e16\u841c\u992e","ting":"\u5385\u542c\u70c3\u6c40\u5ef7\u505c\u4ead\u5ead\u633a\u8247\u839b\u8476\u5a77\u6883\u8713\u9706","tong":"\u901a\u6850\u916e\u77b3\u540c\u94dc\u5f64\u7ae5\u6876\u6345\u7b52\u7edf\u75db\u4f5f\u50ee\u4edd\u833c\u55f5\u6078\u6f7c\u783c","tou":"\u5077\u6295\u5934\u900f\u4ea0","tu":"\u51f8\u79c3\u7a81\u56fe\u5f92\u9014\u6d82\u5c60\u571f\u5410\u5154\u580d\u837c\u83df\u948d\u9174","tuan":"\u6e4d\u56e2\u7583","tui":"\u63a8\u9893\u817f\u8715\u892a\u9000\u5fd2\u717a","tun":"\u541e\u5c6f\u81c0\u9968\u66be\u8c5a\u7a80","tuo":"\u62d6\u6258\u8131\u9e35\u9640\u9a6e\u9a7c\u692d\u59a5\u62d3\u553e\u4e47\u4f57\u5768\u5eb9\u6cb1\u67dd\u7823\u7ba8\u8204\u8dce\u9f0d","wa":"\u6316\u54c7\u86d9\u6d3c\u5a03\u74e6\u889c\u4f64\u5a32\u817d","wai":"\u6b6a\u5916","wan":"\u8c4c\u5f2f\u6e7e\u73a9\u987d\u4e38\u70f7\u5b8c\u7897\u633d\u665a\u7696\u60cb\u5b9b\u5a49\u4e07\u8155\u525c\u8284\u82cb\u83c0\u7ea8\u7efe\u742c\u8118\u7579\u873f\u7ba2","wang":"\u6c6a\u738b\u4ea1\u6789\u7f51\u5f80\u65fa\u671b\u5fd8\u5984\u7f54\u5c22\u60d8\u8f8b\u9b4d","wei":"\u5a01\u5dcd\u5fae\u5371\u97e6\u8fdd\u6845\u56f4\u552f\u60df\u4e3a\u6f4d\u7ef4\u82c7\u840e\u59d4\u4f1f\u4f2a\u5c3e\u7eac\u672a\u851a\u5473\u754f\u80c3\u5582\u9b4f\u4f4d\u6e2d\u8c13\u5c09\u6170\u536b\u502d\u504e\u8bff\u9688\u8473\u8587\u5e0f\u5e37\u5d34\u5d6c\u7325\u732c\u95f1\u6ca9\u6d27\u6da0\u9036\u5a13\u73ae\u97ea\u8ece\u709c\u7168\u71a8\u75ff\u8249\u9c94","wen":"\u761f\u6e29\u868a\u6587\u95fb\u7eb9\u543b\u7a33\u7d0a\u95ee\u520e\u6120\u960c\u6c76\u74ba\u97eb\u6b81\u96ef","weng":"\u55e1\u7fc1\u74ee\u84ca\u8579","wo":"\u631d\u8717\u6da1\u7a9d\u6211\u65a1\u5367\u63e1\u6c83\u83b4\u5e44\u6e25\u674c\u809f\u9f8c","wu":"\u5deb\u545c\u94a8\u4e4c\u6c61\u8bec\u5c4b\u65e0\u829c\u68a7\u543e\u5434\u6bcb\u6b66\u4e94\u6342\u5348\u821e\u4f0d\u4fae\u575e\u620a\u96fe\u6664\u7269\u52ff\u52a1\u609f\u8bef\u5140\u4ef5\u9622\u90ac\u572c\u82b4\u5e91\u6003\u5fe4\u6d6f\u5be4\u8fd5\u59a9\u9a9b\u727e\u7110\u9e49\u9e5c\u8708\u92c8\u9f2f","xi":"\u6614\u7199\u6790\u897f\u7852\u77fd\u6670\u563b\u5438\u9521\u727a\u7a00\u606f\u5e0c\u6089\u819d\u5915\u60dc\u7184\u70ef\u6eaa\u6c50\u7280\u6a84\u88ad\u5e2d\u4e60\u5ab3\u559c\u94e3\u6d17\u7cfb\u9699\u620f\u7ec6\u50d6\u516e\u96b0\u90d7\u831c\u8478\u84f0\u595a\u550f\u5f99\u9969\u960b\u6d60\u6dc5\u5c63\u5b09\u73ba\u6a28\u66e6\u89cb\u6b37\u71b9\u798a\u79a7\u94b8\u7699\u7a78\u8725\u87cb\u823e\u7fb2\u7c9e\u7fd5\u91af\u9f37","xia":"\u778e\u867e\u5323\u971e\u8f96\u6687\u5ce1\u4fa0\u72ed\u4e0b\u53a6\u590f\u5413\u6380\u846d\u55c4\u72ce\u9050\u7455\u7856\u7615\u7f45\u9ee0","xian":"\u9528\u5148\u4ed9\u9c9c\u7ea4\u54b8\u8d24\u8854\u8237\u95f2\u6d8e\u5f26\u5acc\u663e\u9669\u73b0\u732e\u53bf\u817a\u9985\u7fa1\u5baa\u9677\u9650\u7ebf\u51bc\u85d3\u5c98\u7303\u66b9\u5a34\u6c19\u7946\u9e47\u75eb\u86ac\u7b45\u7c7c\u9170\u8df9","xiang":"\u76f8\u53a2\u9576\u9999\u7bb1\u8944\u6e58\u4e61\u7fd4\u7965\u8be6\u60f3\u54cd\u4eab\u9879\u5df7\u6a61\u50cf\u5411\u8c61\u8297\u8459\u9977\u5ea0\u9aa7\u7f03\u87d3\u9c9e\u98e8","xiao":"\u8427\u785d\u9704\u524a\u54ee\u56a3\u9500\u6d88\u5bb5\u6dc6\u6653\u5c0f\u5b5d\u6821\u8096\u5578\u7b11\u6548\u54d3\u54bb\u5d24\u6f47\u900d\u9a81\u7ee1\u67ad\u67b5\u7b71\u7bab\u9b48","xie":"\u6954\u4e9b\u6b47\u874e\u978b\u534f\u631f\u643a\u90aa\u659c\u80c1\u8c10\u5199\u68b0\u5378\u87f9\u61c8\u6cc4\u6cfb\u8c22\u5c51\u5055\u4eb5\u52f0\u71ee\u85a4\u64b7\u5ee8\u7023\u9082\u7ec1\u7f2c\u69ad\u698d\u6b59\u8e9e","xin":"\u85aa\u82af\u950c\u6b23\u8f9b\u65b0\u5ffb\u5fc3\u4fe1\u8845\u56df\u99a8\u8398\u6b46\u94fd\u946b","xing":"\u661f\u8165\u7329\u60fa\u5174\u5211\u578b\u5f62\u90a2\u884c\u9192\u5e78\u674f\u6027\u59d3\u9649\u8347\u8365\u64e4\u60bb\u784e","xiong":"\u5144\u51f6\u80f8\u5308\u6c79\u96c4\u718a\u828e","xiu":"\u4f11\u4fee\u7f9e\u673d\u55c5\u9508\u79c0\u8896\u7ee3\u83a0\u5cab\u9990\u5ea5\u9e3a\u8c85\u9af9","xu":"\u589f\u620c\u9700\u865a\u5618\u987b\u5f90\u8bb8\u84c4\u9157\u53d9\u65ed\u5e8f\u755c\u6064\u7d6e\u5a7f\u7eea\u7eed\u8bb4\u8be9\u5729\u84ff\u6035\u6d2b\u6e86\u987c\u6829\u7166\u7809\u76f1\u80e5\u7cc8\u9191","xuan":"\u8f69\u55a7\u5ba3\u60ac\u65cb\u7384\u9009\u7663\u7729\u7eda\u5107\u8c16\u8431\u63ce\u9994\u6ceb\u6d35\u6e32\u6f29\u7487\u6966\u6684\u70ab\u714a\u78b9\u94c9\u955f\u75c3","xue":"\u9774\u859b\u5b66\u7a74\u96ea\u8840\u5671\u6cf6\u9cd5","xun":"\u52cb\u718f\u5faa\u65ec\u8be2\u5bfb\u9a6f\u5de1\u6b89\u6c5b\u8bad\u8baf\u900a\u8fc5\u5dfd\u57d9\u8340\u85b0\u5ccb\u5f87\u6d54\u66db\u7aa8\u91ba\u9c9f","ya":"\u538b\u62bc\u9e26\u9e2d\u5440\u4e2b\u82bd\u7259\u869c\u5d16\u8859\u6daf\u96c5\u54d1\u4e9a\u8bb6\u4f22\u63e0\u5416\u5c88\u8fd3\u5a05\u740a\u6860\u6c29\u7811\u775a\u75d6","yan":"\u7109\u54bd\u9609\u70df\u6df9\u76d0\u4e25\u7814\u8712\u5ca9\u5ef6\u8a00\u989c\u960e\u708e\u6cbf\u5944\u63a9\u773c\u884d\u6f14\u8273\u5830\u71d5\u538c\u781a\u96c1\u5501\u5f66\u7130\u5bb4\u8c1a\u9a8c\u53a3\u9765\u8d5d\u4fe8\u5043\u5156\u8ba0\u8c33\u90fe\u9122\u82ab\u83f8\u5d26\u6079\u95eb\u960f\u6d07\u6e6e\u6edf\u598d\u5ae3\u7430\u664f\u80ed\u814c\u7131\u7f68\u7b75\u917d\u9b47\u990d\u9f39","yang":"\u6b83\u592e\u9e2f\u79e7\u6768\u626c\u4f6f\u75a1\u7f8a\u6d0b\u9633\u6c27\u4ef0\u75d2\u517b\u6837\u6f3e\u5f89\u600f\u6cf1\u7080\u70ca\u6059\u86d8\u9785","yao":"\u9080\u8170\u5996\u7476\u6447\u5c27\u9065\u7a91\u8c23\u59da\u54ac\u8200\u836f\u8981\u8000\u592d\u723b\u5406\u5d3e\u5fad\u7039\u5e7a\u73e7\u6773\u66dc\u80b4\u9e5e\u7a88\u7e47\u9cd0","ye":"\u6930\u564e\u8036\u7237\u91ce\u51b6\u4e5f\u9875\u6396\u4e1a\u53f6\u66f3\u814b\u591c\u6db2\u8c12\u90ba\u63f6\u9980\u6654\u70e8\u94d8","yi":"\u4e00\u58f9\u533b\u63d6\u94f1\u4f9d\u4f0a\u8863\u9890\u5937\u9057\u79fb\u4eea\u80f0\u7591\u6c82\u5b9c\u59e8\u5f5d\u6905\u8681\u501a\u5df2\u4e59\u77e3\u4ee5\u827a\u6291\u6613\u9091\u5c79\u4ebf\u5f79\u81c6\u9038\u8084\u75ab\u4ea6\u88d4\u610f\u6bc5\u5fc6\u4e49\u76ca\u6ea2\u8be3\u8bae\u8c0a\u8bd1\u5f02\u7ffc\u7fcc\u7ece\u5208\u5293\u4f7e\u8bd2\u572a\u572f\u57f8\u61ff\u82e1\u858f\u5f08\u5955\u6339\u5f0b\u5453\u54a6\u54bf\u566b\u5cc4\u5db7\u7317\u9974\u603f\u6021\u6092\u6f2a\u8fe4\u9a7f\u7f22\u6baa\u8d3b\u65d6\u71a0\u9487\u9552\u9571\u75cd\u7617\u7654\u7fca\u8864\u8734\u8223\u7fbf\u7ff3\u914f\u9edf","yin":"\u8335\u836b\u56e0\u6bb7\u97f3\u9634\u59fb\u541f\u94f6\u6deb\u5bc5\u996e\u5c39\u5f15\u9690\u5370\u80e4\u911e\u5819\u831a\u5591\u72fa\u5924\u6c24\u94df\u763e\u8693\u972a\u9f88","ying":"\u82f1\u6a31\u5a74\u9e70\u5e94\u7f28\u83b9\u8424\u8425\u8367\u8747\u8fce\u8d62\u76c8\u5f71\u9896\u786c\u6620\u5b34\u90e2\u8314\u83ba\u8426\u6484\u5624\u81ba\u6ee2\u6f46\u701b\u745b\u748e\u6979\u9e66\u763f\u988d\u7f42","yo":"\u54df\u5537","yong":"\u62e5\u4f63\u81c3\u75c8\u5eb8\u96cd\u8e0a\u86f9\u548f\u6cf3\u6d8c\u6c38\u607f\u52c7\u7528\u4fd1\u58c5\u5889\u6175\u9095\u955b\u752c\u9cd9\u9954","you":"\u5e7d\u4f18\u60a0\u5fe7\u5c24\u7531\u90ae\u94c0\u72b9\u6cb9\u6e38\u9149\u6709\u53cb\u53f3\u4f51\u91c9\u8bf1\u53c8\u5e7c\u5363\u6538\u4f91\u83b8\u5466\u56ff\u5ba5\u67da\u7337\u7256\u94d5\u75a3\u8763\u9c7f\u9edd\u9f2c","yu":"\u8fc2\u6de4\u4e8e\u76c2\u6986\u865e\u611a\u8206\u4f59\u4fde\u903e\u9c7c\u6109\u6e1d\u6e14\u9685\u4e88\u5a31\u96e8\u4e0e\u5c7f\u79b9\u5b87\u8bed\u7fbd\u7389\u57df\u828b\u90c1\u5401\u9047\u55bb\u5cea\u5fa1\u6108\u6b32\u72f1\u80b2\u8a89\u6d74\u5bd3\u88d5\u9884\u8c6b\u9a6d\u79ba\u6bd3\u4f1b\u4fe3\u8c00\u8c15\u8438\u84e3\u63c4\u5581\u5704\u5709\u5d5b\u72f3\u996b\u5ebe\u9608\u59aa\u59a4\u7ea1\u745c\u6631\u89ce\u8174\u6b24\u65bc\u715c\u71e0\u807f\u94b0\u9e46\u7610\u7600\u7ab3\u8753\u7afd\u8201\u96e9\u9f89","yuan":"\u9e33\u6e0a\u51a4\u5143\u57a3\u8881\u539f\u63f4\u8f95\u56ed\u5458\u5706\u733f\u6e90\u7f18\u8fdc\u82d1\u613f\u6028\u9662\u586c\u6c85\u5a9b\u7457\u6a7c\u7230\u7722\u9e22\u8788\u9f0b","yue":"\u66f0\u7ea6\u8d8a\u8dc3\u94a5\u5cb3\u7ca4\u6708\u60a6\u9605\u9fa0\u6a3e\u5216\u94ba","yun":"\u8018\u4e91\u90e7\u5300\u9668\u5141\u8fd0\u8574\u915d\u6655\u97f5\u5b55\u90d3\u82b8\u72c1\u607d\u7ead\u6b92\u6600\u6c32","za":"\u531d\u7838\u6742\u62f6\u5482","zai":"\u683d\u54c9\u707e\u5bb0\u8f7d\u518d\u5728\u54b1\u5d3d\u753e","zan":"\u6512\u6682\u8d5e\u74d2\u661d\u7c2a\u7ccc\u8db1\u933e","zang":"\u8d43\u810f\u846c\u5958\u6215\u81e7","zao":"\u906d\u7cdf\u51ff\u85fb\u67a3\u65e9\u6fa1\u86a4\u8e81\u566a\u9020\u7682\u7076\u71e5\u5523\u7f2b","ze":"\u8d23\u62e9\u5219\u6cfd\u4ec4\u8d5c\u5567\u8fee\u6603\u7b2e\u7ba6\u8234","zei":"\u8d3c","zen":"\u600e\u8c2e","zeng":"\u589e\u618e\u66fe\u8d60\u7f2f\u7511\u7f7e\u9503","zha":"\u624e\u55b3\u6e23\u672d\u8f67\u94e1\u95f8\u7728\u6805\u69a8\u548b\u4e4d\u70b8\u8bc8\u63f8\u5412\u54a4\u54f3\u600d\u781f\u75c4\u86b1\u9f44","zhai":"\u6458\u658b\u5b85\u7a84\u503a\u5be8\u7826","zhan":"\u77bb\u6be1\u8a79\u7c98\u6cbe\u76cf\u65a9\u8f97\u5d2d\u5c55\u8638\u6808\u5360\u6218\u7ad9\u6e5b\u7efd\u8c35\u640c\u65c3","zhang":"\u6a1f\u7ae0\u5f70\u6f33\u5f20\u638c\u6da8\u6756\u4e08\u5e10\u8d26\u4ed7\u80c0\u7634\u969c\u4ec9\u9123\u5e5b\u5d82\u7350\u5adc\u748b\u87d1","zhao":"\u62db\u662d\u627e\u6cbc\u8d75\u7167\u7f69\u5146\u8087\u53ec\u722a\u8bcf\u68f9\u948a\u7b0a","zhe":"\u906e\u6298\u54f2\u86f0\u8f99\u8005\u9517\u8517\u8fd9\u6d59\u8c2a\u966c\u67d8\u8f84\u78d4\u9e67\u891a\u8707\u8d6d","zhen":"\u73cd\u659f\u771f\u7504\u7827\u81fb\u8d1e\u9488\u4fa6\u6795\u75b9\u8bca\u9707\u632f\u9547\u9635\u7f1c\u6862\u699b\u8f78\u8d48\u80d7\u6715\u796f\u755b\u9e29","zheng":"\u84b8\u6323\u7741\u5f81\u72f0\u4e89\u6014\u6574\u62ef\u6b63\u653f\u5e27\u75c7\u90d1\u8bc1\u8be4\u5ce5\u94b2\u94ee\u7b5d","zhi":"\u829d\u679d\u652f\u5431\u8718\u77e5\u80a2\u8102\u6c41\u4e4b\u7ec7\u804c\u76f4\u690d\u6b96\u6267\u503c\u4f84\u5740\u6307\u6b62\u8dbe\u53ea\u65e8\u7eb8\u5fd7\u631a\u63b7\u81f3\u81f4\u7f6e\u5e1c\u5cd9\u5236\u667a\u79e9\u7a1a\u8d28\u7099\u75d4\u6ede\u6cbb\u7a92\u536e\u965f\u90c5\u57f4\u82b7\u646d\u5e19\u5fee\u5f58\u54ab\u9a98\u6809\u67b3\u6800\u684e\u8f75\u8f7e\u6534\u8d3d\u81a3\u7949\u7957\u9ef9\u96c9\u9e37\u75e3\u86ed\u7d77\u916f\u8dd6\u8e2c\u8e2f\u8c78\u89ef","zhong":"\u4e2d\u76c5\u5fe0\u949f\u8877\u7ec8\u79cd\u80bf\u91cd\u4ef2\u4f17\u51a2\u953a\u87bd\u8202\u822f\u8e35","zhou":"\u821f\u5468\u5dde\u6d32\u8bcc\u7ca5\u8f74\u8098\u5e1a\u5492\u76b1\u5b99\u663c\u9aa4\u5544\u7740\u501c\u8bf9\u836e\u9b3b\u7ea3\u80c4\u78a1\u7c40\u8233\u914e\u9cb7","zhu":"\u73e0\u682a\u86db\u6731\u732a\u8bf8\u8bdb\u9010\u7af9\u70db\u716e\u62c4\u77a9\u5631\u4e3b\u8457\u67f1\u52a9\u86c0\u8d2e\u94f8\u7b51\u4f4f\u6ce8\u795d\u9a7b\u4f2b\u4f8f\u90be\u82ce\u8331\u6d19\u6e1a\u6f74\u9a7a\u677c\u69e0\u6a65\u70b7\u94e2\u75b0\u7603\u86b0\u7afa\u7bb8\u7fe5\u8e85\u9e88","zhua":"\u6293","zhuai":"\u62fd","zhuan":"\u4e13\u7816\u8f6c\u64b0\u8d5a\u7bc6\u629f\u556d\u989b","zhuang":"\u6869\u5e84\u88c5\u5986\u649e\u58ee\u72b6\u4e2c","zhui":"\u690e\u9525\u8ffd\u8d58\u5760\u7f00\u8411\u9a93\u7f12","zhun":"\u8c06\u51c6","zhuo":"\u6349\u62d9\u5353\u684c\u7422\u8301\u914c\u707c\u6d4a\u502c\u8bfc\u5ef4\u855e\u64e2\u555c\u6d5e\u6dbf\u6753\u712f\u799a\u65ab","zi":"\u5179\u54a8\u8d44\u59ff\u6ecb\u6dc4\u5b5c\u7d2b\u4ed4\u7c7d\u6ed3\u5b50\u81ea\u6e0d\u5b57\u8c18\u5d6b\u59ca\u5b73\u7f01\u6893\u8f8e\u8d40\u6063\u7726\u9531\u79ed\u8014\u7b2b\u7ca2\u89dc\u8a3e\u9cbb\u9aed","zong":"\u9b03\u68d5\u8e2a\u5b97\u7efc\u603b\u7eb5\u8159\u7cbd","zou":"\u90b9\u8d70\u594f\u63cd\u9139\u9cb0","zu":"\u79df\u8db3\u5352\u65cf\u7956\u8bc5\u963b\u7ec4\u4fce\u83f9\u5550\u5f82\u9a75\u8e74","zuan":"\u94bb\u7e82\u6525\u7f35","zui":"\u5634\u9189\u6700\u7f6a","zun":"\u5c0a\u9075\u6499\u6a3d\u9cdf","zuo":"\u6628\u5de6\u4f50\u67de\u505a\u4f5c\u5750\u5ea7\u961d\u963c\u80d9\u795a\u9162","cou":"\u85ae\u6971\u8f8f\u8160","nang":"\u652e\u54dd\u56d4\u9995\u66e9","o":"\u5594","dia":"\u55f2","chuai":"\u562c\u81aa\u8e39","cen":"\u5c91\u6d94","diu":"\u94e5","nou":"\u8028","fou":"\u7f36","bia":"\u9adf"};
function Cn2PinYin(l1) {
    var l2 = l1.length;
    var I1 = "";
    var reg = new RegExp('[a-zA-Z0-9\- ]');
    for (var i=0;i<l2;i++) {
        var val = l1.substr(i,1);
        var name = arraySearch(val,PinYin);
        if(reg.test(val)) {
            I1 += val;
        } else if (name!==false) {
            I1 += name;
        }
    }
    I1 = I1.replace(/ /g,'-');
    while (I1.indexOf('--')>0) {
        I1 = I1.replace('--','-');
    }
    return I1;
}

function arraySearch(l1,l2){
    for (var name in PinYin){
        if (PinYin[name].indexOf(l1)!=-1) {
            return name; break;
        }
    }
    return false;
}

function get_cur_pinyin(content)
{
    var fn = Cn2PinYin;
    var arr = [];
    var ch;
    for(var i=0,n=content.length; i<n; i++)
    {
        ch = content.charAt(i);
        arr[i] = (ch<"一" || ch>"龥")?ch:fn(ch);
    }
    return arr.join("");
}
var AutoSiteSearch={
    search_url:"http://cgi.data.auto.qq.com/php/search.php",
    auto_url:"http://data.auto.qq.com/",
    cur_keyword:"",
    is_presskeyword:false,
    init_keyword_str:"车型名称/拼音",
    ad_keyword_str:"车型名称/拼音",
    all_inputlist_num:0,
    cur_inputlist_num:0,
    parameter:function(name){
        var reg=new RegExp("([\?|&])"+ name +"=([^&]*)");
        if (reg.test(location.href)) return decodeURIComponent(RegExp.$2.replace(/\+/g, " ")); return "";
    },
    $:function(s){
        return(typeof s=='object')?s:document.getElementById(s);
    },
    bind:function(el,evt,fn){
        if(el.attachEvent){
            el.attachEvent("on"+evt,fn);
        }else if(el.addEventListener){
            el.addEventListener(evt,fn,false);
        }else{
            el["on"+evt]=fn;
        }
    },//$(s)
    Browser:{

        ie:/msie/.test(window.navigator.userAgent.toLowerCase()),
        moz:/gecko/.test(window.navigator.userAgent.toLowerCase()),
        opera:/opera/.test(window.navigator.userAgent.toLowerCase()),
        safari:/safari/.test(window.navigator.userAgent.toLowerCase())

    },

    JsLoader:{
        load: function(sUrl, fCallback, chset)
        {
            var _script = document.createElement("script");
            _script.setAttribute("type", "text/javascript");
            _script.setAttribute("src", sUrl);

            if (chset)
                _script.setAttribute("charset", chset);
            else
                _script.setAttribute("charset", "gb2312");

            document.getElementsByTagName("head")[0].appendChild(_script);

            if (AutoSiteSearch.Browser.ie)
            {
                _script.onreadystatechange = function()
                {
                    if (this.readyState=="loaded" || this.readyState=="complete")
                    {
                        fCallback();
                    }
                };
            }
            else if (AutoSiteSearch.Browser.moz)
            {
                _script.onload = function()
                {
                    fCallback();
                };
            }
            else
            {
                fCallback();
            }
        }
    },
    is_InArray:function(arr,val)
    {
        for(var i=0;i<arr.length;i++)
        {
            if(arr[i] == val)
                return true;
        }
        return false;
    },
    clickSearchButton:function(){
        AutoSiteSearch.cur_inputlist_num=0;
        return AutoSiteSearch.searchByKeyword();
    },
    searchByKeyword:function(){
        var keyword = document.getElementById("id_keyword").value;
        var cata = 0;
        if (keyword == null || keyword == "" || keyword == AutoSiteSearch.init_keyword_str)
        {
            alert("\u8BF7\u8F93\u5165\u67E5\u8BE2\u5173\u952E\u5B57");
            return false;
        }
        switch (parseInt(cata)) {
            case 0:
            {
                if(AutoSiteSearch.cur_inputlist_num!=0)
                {
                    AutoSiteSearch.search_keyword_v2(AutoSiteSearch.$('input_a_'+AutoSiteSearch.cur_inputlist_num).innerHTML);
                    return false;
                }

                BossLinkAutoSearch(32,0,encodeURIComponent(keyword)); //点击了关键字搜索按钮

                window.open("http://cgi.data.auto.qq.com/php/search.php?fuzzy_search=1&keyword=" + encodeURIComponent(keyword));
            }
                break;
            case 1:
                window.open("http://news.soso.com/n.q?w=" + keyword + "&site=auto.qq.com&cid=w.q.auto.s&ty=diary&sc=news");
                break;
            case 2:
                window.open("http://blog.soso.com/qz.q?w=" + keyword + "&site=auto.qq.com&cid=w.q.auto.s&ty=diary&sc=blog");
                break;
            case 3:
                window.open("http://bbs.qq.com/search/?cont=" + keyword);
                break;
            default:;
        }
        return false;
    },//searchByKeyword()
    keyup_keyword:function(e)
    {
        var ev = e? e : window.event;
        AutoSiteSearch.is_presskeyword=true;
        if(ev.keyCode==27)
        {
            AutoSiteSearch.blur_keyword(AutoSiteSearch.$('id_keyword'));
            AutoSiteSearch.close_smartbox();
        }else if(ev.keyCode==40)
        {
            //方向键-下
            if(AutoSiteSearch.$('id_keyword').value=='')
            {
                return false;
            }
            AutoSiteSearch.$('search_inputlist').style.display="block";
            AutoSiteSearch.focus_inputlist(AutoSiteSearch.cur_inputlist_num,0);
        }else if(ev.keyCode==38){
            //方向键-上
            if(AutoSiteSearch.$('id_keyword').value=='')
            {
                return false;
            }
            AutoSiteSearch.$('search_inputlist').style.display="block";
            AutoSiteSearch.focus_inputlist(AutoSiteSearch.cur_inputlist_num,1);
        }else{
            AutoSiteSearch.open_smartbox();
        }
    },
    focus_inputlist:function(id,flag)
    {
        if(flag==0)
        {
            id+=1;
        }else{
            id-=1;
        }
        if(id>=AutoSiteSearch.all_inputlist_num)
        {
            id=AutoSiteSearch.all_inputlist_num;
        }
        if(id<=0)
        {
            id=0;
        }
        if(AutoSiteSearch.$('input_item_'+id))
        {
            for(i=1;i<(AutoSiteSearch.all_inputlist_num+1);i++)
            {
                AutoSiteSearch.$('input_item_'+i).className='';
            }
            AutoSiteSearch.$('input_item_'+id).className="focus2";
            AutoSiteSearch.cur_inputlist_num=id;
        }
    },
    open_smartbox:function()
    {
        var keyword=AutoSiteSearch.$('id_keyword').value;
        AutoSiteSearch.search_smartbox(keyword);
        return false;
    },
    close_smartbox:function()
    {
        AutoSiteSearch.cur_inputlist_num=0;
        AutoSiteSearch.close_single_smartbox();
        return false;
        alert("sony close");
    },
    close_single_smartbox:function()
    {
        var item=AutoSiteSearch.$('search_inputlist');
        if(typeof(item)!=null && item.style.display != 'none')
        {
            item.style.display="none";
        }
        return true;
    },
    search_smartbox:function(keyword)
    {
        AutoSiteSearch.all_inputlist_num=0;
        AutoSiteSearch.cur_inputlist_num=0;
        if(keyword!='' && keyword!=AutoSiteSearch.init_keyword_str)
        {
            //先转换为拼音
            keyword=get_cur_pinyin(keyword.toLowerCase().replace(/\s/,''));
            //再进行比对
            var tmp_html_str='<ul>';
            if(typeof _autoapp_site_serial_py_json != "undefined")
            {
                for(key in _autoapp_site_serial_py_json)
                {
                    if(AutoSiteSearch.all_inputlist_num>=9)
                    {
                        break;
                    }
                    if(_autoapp_site_serial_py_json[key]['p'].match(keyword)!=null)
                    {
                        tmp_html_str+='<li id="input_item_'+(AutoSiteSearch.all_inputlist_num+1)+'" onmouseover="return AutoSiteSearch.mouseover_item('+(AutoSiteSearch.all_inputlist_num+1)+');"><a href="#" id="input_a_'+(AutoSiteSearch.all_inputlist_num+1)+'" style="display:none">'+_autoapp_site_serial_py_json[key]['i']+'</a><a href="javascript:alert(1);" onclick="return AutoSiteSearch.click_inputlist_item('+(AutoSiteSearch.all_inputlist_num+1)+','+_autoapp_site_serial_py_json[key]['i']+');" title="'+_autoapp_site_serial_py_json[key]['n']+'">';
                        tmp_sname=_autoapp_site_serial_py_json[key]['n'];
                        if(AutoSiteSearch.gb_length(tmp_sname)>20)
                        {
                            tmp_sname=AutoSiteSearch.gb_substr(tmp_sname,17)+"...";
                        }

                        tmp_html_str+=tmp_sname;
                        if(typeof(_autoapp_site_serial_py_json[key]['d'])!='undefined')
                        {
                            tmp_html_str+='<font style="color:#666">(进口)</font>';
                            tmp_sname+='(进口)';
                        }
                        tmp_sblank='';
                        for(i=AutoSiteSearch.gb_length(tmp_sname);i<=23;i++)
                        {
                            tmp_sblank+='&nbsp;';
                        }
                        tmp_ss=_autoapp_site_serial_py_json[key]['s'].toUpperCase();
                        tmp_sk=keyword.toUpperCase();
                        tmp_pos=tmp_ss.indexOf(tmp_sk);
                        if(tmp_pos>=0)
                        {
                            tmp_ss0=tmp_ss.substr(0,tmp_pos);
                            tmp_ss1=tmp_sk;
                            tmp_ss2=tmp_ss.substr(tmp_pos+tmp_ss1.length);
                        }
                        else
                        {
                            tmp_ss0=tmp_ss;
                            tmp_ss1='';
                            tmp_ss2='';
                        }
                        //tmp_html_str+=tmp_sblank+'<font style="color:red">'+_autoapp_site_serial_py_json[key]['s'].toUpperCase()+'</font>';
                        tmp_html_str+=tmp_sblank+'<font style="color:#333333;">'+tmp_ss0+'</font><font style="color:#d90606">'+tmp_ss1+'</font><font style="color:#333333">'+tmp_ss2+'</font>';
                        tmp_html_str+='</a>';
                        tmp_html_str+='</li>';
                        AutoSiteSearch.all_inputlist_num++;
                    }
                }
                if(AutoSiteSearch.all_inputlist_num==0)
                {
                    tmp_html_str+='无相关推荐';
                }
            }
            tmp_html_str+='</ul>';
            AutoSiteSearch.$('search_inputlist').innerHTML=tmp_html_str;
            AutoSiteSearch.$('search_inputlist').style.display="block";
        }else{
            AutoSiteSearch.close_smartbox();
        }
    },
    mouseover_item:function(id)
    {
        for(i=1;i<(AutoSiteSearch.all_inputlist_num+1);i++)
        {
            AutoSiteSearch.$('input_item_'+i).className='';
        }
        AutoSiteSearch.$('input_item_'+id).className='focus2';
        AutoSiteSearch.cur_inputlist_num=id;
        return false;
    },
    click_inputlist_item:function(item,id){
        AutoSiteSearch.cur_inputlist_num=item;
        for(i=1;i<(AutoSiteSearch.all_inputlist_num+1);i++)
        {
            AutoSiteSearch.$('input_item_'+i).className='';
        }
        AutoSiteSearch.close_smartbox();
        AutoSiteSearch.search_keyword_v2(id);
        return false;
    },
    search_keyword_v2:function(id)
    {
        var cur_id=parseInt(id,10);
        if(isNaN(cur_id) || cur_id==0)
        {
            return false;
        }

        BossLinkAutoSearch(31,cur_id,""); //关键字搜索中选择了某个推荐车系

        var search_tmp_str=AutoSiteSearch.auto_url+"car_serial/"+cur_id+"/index.shtml";
        window.open(search_tmp_str);
        return false;
    },
    blur_keyword:function(item)
    {
        if(item.value=='')
        {
            item.value=AutoSiteSearch.ad_keyword_str;
        }
        if(item.value==AutoSiteSearch.ad_keyword_str||item.value=='')
        {
            item.style.color='gray';
        }
        return true;
    },
    init_py_list:function(){
        var py_list_url="http://js.data.auto.qq.com/car_public/template/serial_py.js";
        AutoSiteSearch.JsLoader.load(py_list_url,function(){
            if(typeof(_autoapp_site_serial_py_json)!="undefined")
            {
                //如果正常调入，那么执行转换拼音函数初始化功能。
                //set_pinyin_init();
            }
        });
    },
    cancel_event:function(evt)
    {
        evt=evt||window.event;
        if(evt.preventDefault){
            evt.stopPropagation();
            evt.preventDefault();
        }else{
            evt.returnValue=false;
            evt.cancelBubble=true;
        }
    },
    click_keyword:function(obj,evt)
    {
        BossLinkAutoSearch(30,0,""); //点击了关键字搜索框

        AutoSiteSearch.cancel_event(evt);
        if(obj.value==AutoSiteSearch.ad_keyword_str)
        {
            obj.value='';
            obj.style.color="black";
        }
        AutoSiteSearch.open_smartbox();
        return false;
    },
    gb_substr:function(str,len)
    {
        if (len == null || len <= 0)
        {
            return "";
        }
        var str_len = str.length;
        if(len >= str_len * 2)
        {
            return str;
        }
        var get_len = 0;
        for (;get_len < str_len && len > 0; get_len++)
        {
            if (str.charCodeAt(get_len) > 255 || str.charCodeAt(get_len) < 0)
            {
                len -= 2;
            }else{
                len--;
            }
        }
        if (len < 0)
        {
            get_len -= 1;
        }
        if (get_len == str_len)
        {
            return str;
        }else{
            return str.substr(0,get_len);
        }
    },
    gb_length:function(str)
    {
        return str.replace(/[^\x00-\xff]/gi,'xx').length;
    }
};

//订阅板块过滤?字符
function dyReplace(){
    $("#dy a").each(function(){$(this).html($(this).html().replace("?",""))});
}

//娱乐个性化灰度
function entPersonal(){
    if($("#ent .contentRight").is(":hidden")){
    }else{
        var uinStr = QQ.Cookie.get("uin") || QQ.Cookie.get("luin") || QQ.Cookie.get("pt2qquin") || QQ.Cookie.get("o_cookie") || "";
        uinStr = parseInt(Number(uinStr.replace("o",""))).toString();
        /*
        if(window.location.toString().indexOf('qq=') != -1){
            var qqTest = window.location.toString().substring(window.location.toString().indexOf('qq=')+3);
            uinStr = qqTest;
        }
        var uinBetaArr = ["57166400","517310403","270961575","532745901","69311328","14497563","1211042752","534665556","1160153028","434553806","960289793","1092603568","1254908228","2993768685","805768748","1587194124","840041635","6612201","393002292","1137247079","1656379013","2298753329","1142702209","690054798","403863641","89219370","405676994"];
        var uinBetaArrLen = uinBetaArr.length;
        for(var i=0;i<uinBetaArrLen;i++){ //白名单
            if(uinStr == uinBetaArr[i]){
                QQ.LoadScript("http://i.match.qq.com/newalgorithm/groupnews?callback=entCallback&qq="+uinStr+"&channel=ent&ran="+Math.random(),function(){
                    $("#ent .contentRight").attr("bossZone","ent_2");
                });
                return false;
            }
        }
        if(uinStr.charAt(uinStr.length-3) <= 7){ //娱乐个性化
        }
        */
        QQ.LoadScript("http://i.match.qq.com/newalgorithm/groupnews?callback=entCallback&channel=ent&ran="+Math.random(),function(){
            $("#ent .contentRight").attr("bossZone","ent_2");
            if(uinStr.charAt(uinStr.length-3) == 0){
                $("#ent .contentRight").attr("bossZone","ent_5");
            }
        });
    }
}

//时尚个性化灰度
function fashionPersonal(){
    if($("#fashion .contentRight").is(":hidden")){
    	var fold = 0;
    }else{
    	var fold = 1;
    }
    var uinStr = QQ.Cookie.get("uin") || QQ.Cookie.get("luin") || QQ.Cookie.get("pt2qquin") || QQ.Cookie.get("o_cookie") || "";
    uinStr = parseInt(Number(uinStr.replace("o",""))).toString();
    /*
    if(window.location.toString().indexOf('qq=') != -1){
        var qqTest = window.location.toString().substring(window.location.toString().indexOf('qq=')+3);
        uinStr = qqTest;
    }
    var uinBetaArr = ["1587194124","2202051225","254823998","57166400","517310403","89219370"];
    var uinBetaArrLen = uinBetaArr.length;
    for(var i=0;i<uinBetaArrLen;i++){ //白名单
        if(uinStr == uinBetaArr[i]){
            QQ.LoadScript("http://i.match.qq.com/newalgorithm/groupnews?callback=fashionCallback&qq="+uinStr+"&channel=fashion&fold="+fold+"&ran="+Math.random(),function(){
                $("#fashion .contentRight").attr("bossZone","fashion_2");
            });
            return false;
        }
    }
    */
  	if(uinStr.charAt(uinStr.length-3)%2 == 0){ //时尚个性化
        QQ.LoadScript("http://i.match.qq.com/newalgorithm/groupnews?callback=fashionCallback&qq="+uinStr+"&channel=fashion&fold="+fold+"&ran="+Math.random(),function(){
            $("#fashion .contentRight").attr("bossZone","fashion_2");
        });
  	}
}

function entCallback(obj){
    if(obj.code ==0){
        $("#ent .contentLeft .imgArea").html('<a target="_blank" href="'+obj.data[0].url+'"><img src="'+obj.data[0].img+'" alt="'+obj.data[0].title+'" /></a>');
        $("#ent .contentLeft .txtArea h3").html('<a target="_blank" href="'+obj.data[0].url+'">'+obj.data[0].title+'</a>');
        var entLeftDataArr = [];
        for(var i=1;i<7;i++){
            if(obj.data[i].is_locked && obj.data[i].is_locked == "2"){
                entLeftDataArr.push('<li>');
                for(var n=0; n<obj.data[i].items.length; n++){
                    entLeftDataArr.push('<a class="');
                    if(obj.data[i].items[n].ext){
                        if(obj.data[i].items[n].ext.style.bold == "1"){ //加粗
                            entLeftDataArr.push('Q-bold ');
                        }
                        if(obj.data[i].items[n].ext.style.color == "1"){ //红色
                            entLeftDataArr.push('qm_c_1 ');
                        }else if(obj.data[i].items[n].ext.style.color == "2"){ //黑色
                            entLeftDataArr.push('qm_c_2 ');
                        }else if(obj.data[i].items[n].ext.style.color == "3"){ //蓝色
                            entLeftDataArr.push('qm_c_3 ');
                        }
                        if(obj.data[i].items[n].ext.style.icon == "1"){ //微博图标
                            entLeftDataArr.push('qm_i_1');
                        }else if(obj.data[i].items[n].ext.style.icon == "2"){ //视频图标
                            entLeftDataArr.push('qm_i_2');
                        }else if(obj.data[i].items[n].ext.style.icon == "3"){ //图片图标
                            entLeftDataArr.push('qm_i_3');
                        }else if(obj.data[i].items[n].ext.style.icon == "4"){ //直播图标
                            entLeftDataArr.push('qm_i_4');
                        }else if(obj.data[i].items[n].ext.style.icon == "5"){ //声音图标
                            entLeftDataArr.push('audioico');
                        }
                    }
                    entLeftDataArr.push('" target="_blank" href="'+obj.data[i].items[n].url+'">'+obj.data[i].items[n].title+'</a>');
                    if(obj.data[i].items[n].ext){ //竖线
                        if(obj.data[i].items[n].ext.split == "1"){ //竖线
                            entLeftDataArr.push(' | ');
                        }else{
                            entLeftDataArr.push(' ');
                        }
                    }else{
                        entLeftDataArr.push(' ');
                    }
                }
                entLeftDataArr.push('</li>');
            }else{
                entLeftDataArr.push('<li><a class="');
                if(obj.data[i].ext){
                    if(obj.data[i].ext.style.bold == "1"){ //加粗
                        entLeftDataArr.push('Q-bold ');
                    }
                    if(obj.data[i].ext.style.color == "1"){ //红色
                        entLeftDataArr.push('qm_c_1 ');
                    }else if(obj.data[i].ext.style.color == "2"){ //黑色
                        entLeftDataArr.push('qm_c_2 ');
                    }else if(obj.data[i].ext.style.color == "3"){ //蓝色
                        entLeftDataArr.push('qm_c_3 ');
                    }
                    if(obj.data[i].ext.style.icon == "1"){ //微博图标
                        entLeftDataArr.push('qm_i_1');
                    }else if(obj.data[i].ext.style.icon == "2"){ //视频图标
                        entLeftDataArr.push('qm_i_2');
                    }else if(obj.data[i].ext.style.icon == "3"){ //图片图标
                        entLeftDataArr.push('qm_i_3');
                    }else if(obj.data[i].ext.style.icon == "4"){ //直播图标
                        entLeftDataArr.push('qm_i_4');
                    }else if(obj.data[i].ext.style.icon == "5"){ //声音图标
                        entLeftDataArr.push('audioico');
                    }
                }
                entLeftDataArr.push('" target="_blank" href="'+obj.data[i].url+'">'+obj.data[i].title+'</a></li>');
            }
        }
        $("#ent .contentLeft ul").html(entLeftDataArr.join(""));

        var entRightDataArr = [];
        for(var i=7;i<16;i++){
            if(obj.data[i].is_locked && obj.data[i].is_locked == "2"){
                entRightDataArr.push('<li>');
                for(var n=0; n<obj.data[i].items.length; n++){
                    entRightDataArr.push('<a class="');
                    if(obj.data[i].items[n].ext){
                        if(obj.data[i].items[n].ext.style.bold == "1"){ //加粗
                            entRightDataArr.push('Q-bold ');
                        }
                        if(obj.data[i].items[n].ext.style.color == "1"){ //红色
                            entRightDataArr.push('qm_c_1 ');
                        }else if(obj.data[i].items[n].ext.style.color == "2"){ //黑色
                            entRightDataArr.push('qm_c_2 ');
                        }else if(obj.data[i].items[n].ext.style.color == "3"){ //蓝色
                            entRightDataArr.push('qm_c_3 ');
                        }
                        if(obj.data[i].items[n].ext.style.icon == "1"){ //微博图标
                            entRightDataArr.push('qm_i_1');
                        }else if(obj.data[i].items[n].ext.style.icon == "2"){ //视频图标
                            entRightDataArr.push('qm_i_2');
                        }else if(obj.data[i].items[n].ext.style.icon == "3"){ //图片图标
                            entRightDataArr.push('qm_i_3');
                        }else if(obj.data[i].items[n].ext.style.icon == "4"){ //直播图标
                            entRightDataArr.push('qm_i_4');
                        }else if(obj.data[i].items[n].ext.style.icon == "5"){ //声音图标
                            entRightDataArr.push('audioico');
                        }
                    }
                    entRightDataArr.push('" target="_blank" href="'+obj.data[i].items[n].url+'">'+obj.data[i].items[n].title+'</a>');
                    if(obj.data[i].items[n].ext){ //竖线
                        if(obj.data[i].items[n].ext.split == "1"){ //竖线
                            entRightDataArr.push(' | ');
                        }else{
                            entRightDataArr.push(' ');
                        }
                    }else{
                        entRightDataArr.push(' ');
                    }
                }
                entRightDataArr.push('</li>');
            }else{
                entRightDataArr.push('<li><a class="');
                if(obj.data[i].ext){
                    if(obj.data[i].ext.style.bold == "1"){ //加粗
                        entRightDataArr.push('Q-bold ');
                    }
                    if(obj.data[i].ext.style.color == "1"){ //红色
                        entRightDataArr.push('qm_c_1 ');
                    }else if(obj.data[i].ext.style.color == "2"){ //黑色
                        entRightDataArr.push('qm_c_2 ');
                    }else if(obj.data[i].ext.style.color == "3"){ //蓝色
                        entRightDataArr.push('qm_c_3 ');
                    }
                    if(obj.data[i].ext.style.icon == "1"){ //微博图标
                        entRightDataArr.push('qm_i_1');
                    }else if(obj.data[i].ext.style.icon == "2"){ //视频图标
                        entRightDataArr.push('qm_i_2');
                    }else if(obj.data[i].ext.style.icon == "3"){ //图片图标
                        entRightDataArr.push('qm_i_3');
                    }else if(obj.data[i].ext.style.icon == "4"){ //直播图标
                        entRightDataArr.push('qm_i_4');
                    }else if(obj.data[i].ext.style.icon == "5"){ //声音图标
                        entRightDataArr.push('audioico');
                    }
                }
                entRightDataArr.push('" target="_blank" href="'+obj.data[i].url+'">'+obj.data[i].title+'</a></li>');
            }
        }
        $("#ent .contentRight ul").html(entRightDataArr.join(""));
    }
}

function fashionCallback(obj){
    if(obj.code ==0){
        $("#fashion .contentLeft .imgArea").html('<a target="_blank" href="'+obj.data[0].url+'"><img src="'+obj.data[0].img+'" alt="'+obj.data[0].title+'" /></a>');
        $("#fashion .contentLeft .txtArea h3").html('<a target="_blank" href="'+obj.data[0].url+'">'+obj.data[0].title+'</a>');
        var fashionLeftDataArr = [];
        for(var i=1;i<7;i++){
            if(obj.data[i].is_locked && obj.data[i].is_locked == "2"){
                fashionLeftDataArr.push('<li>');
                for(var n=0; n<obj.data[i].items.length; n++){
                    fashionLeftDataArr.push('<a class="');
                    if(obj.data[i].items[n].ext){
                        if(obj.data[i].items[n].ext.style.bold == "1"){ //加粗
                            fashionLeftDataArr.push('Q-bold ');
                        }
                        if(obj.data[i].items[n].ext.style.color == "1"){ //红色
                            fashionLeftDataArr.push('qm_c_1 ');
                        }else if(obj.data[i].items[n].ext.style.color == "2"){ //黑色
                            fashionLeftDataArr.push('qm_c_2 ');
                        }else if(obj.data[i].items[n].ext.style.color == "3"){ //蓝色
                            fashionLeftDataArr.push('qm_c_3 ');
                        }
                        if(obj.data[i].items[n].ext.style.icon == "1"){ //微博图标
                            fashionLeftDataArr.push('qm_i_1');
                        }else if(obj.data[i].items[n].ext.style.icon == "2"){ //视频图标
                            fashionLeftDataArr.push('qm_i_2');
                        }else if(obj.data[i].items[n].ext.style.icon == "3"){ //图片图标
                            fashionLeftDataArr.push('qm_i_3');
                        }else if(obj.data[i].items[n].ext.style.icon == "4"){ //直播图标
                            fashionLeftDataArr.push('qm_i_4');
                        }else if(obj.data[i].items[n].ext.style.icon == "5"){ //声音图标
                            fashionLeftDataArr.push('audioico');
                        }
                    }
                    fashionLeftDataArr.push('" target="_blank" href="'+obj.data[i].items[n].url+'">'+obj.data[i].items[n].title+'</a>');
                    if(obj.data[i].items[n].ext){ //竖线
                        if(obj.data[i].items[n].ext.split == "1"){ //竖线
                            fashionLeftDataArr.push(' | ');
                        }else{
                            fashionLeftDataArr.push(' ');
                        }
                    }else{
                        fashionLeftDataArr.push(' ');
                    }
                }
                fashionLeftDataArr.push('</li>');
            }else{
                fashionLeftDataArr.push('<li><a class="');
                if(obj.data[i].ext){
                    if(obj.data[i].ext.style.bold == "1"){ //加粗
                        fashionLeftDataArr.push('Q-bold ');
                    }
                    if(obj.data[i].ext.style.color == "1"){ //红色
                        fashionLeftDataArr.push('qm_c_1 ');
                    }else if(obj.data[i].ext.style.color == "2"){ //黑色
                        fashionLeftDataArr.push('qm_c_2 ');
                    }else if(obj.data[i].ext.style.color == "3"){ //蓝色
                        fashionLeftDataArr.push('qm_c_3 ');
                    }
                    if(obj.data[i].ext.style.icon == "1"){ //微博图标
                        fashionLeftDataArr.push('qm_i_1');
                    }else if(obj.data[i].ext.style.icon == "2"){ //视频图标
                        fashionLeftDataArr.push('qm_i_2');
                    }else if(obj.data[i].ext.style.icon == "3"){ //图片图标
                        fashionLeftDataArr.push('qm_i_3');
                    }else if(obj.data[i].ext.style.icon == "4"){ //直播图标
                        fashionLeftDataArr.push('qm_i_4');
                    }else if(obj.data[i].ext.style.icon == "5"){ //声音图标
                        fashionLeftDataArr.push('audioico');
                    }
                }
                fashionLeftDataArr.push('" target="_blank" href="'+obj.data[i].url+'">'+obj.data[i].title+'</a></li>');
            }
        }
        $("#fashion .contentLeft ul").html(fashionLeftDataArr.join(""));
      
		if($("#fashion .contentRight").is(":hidden")){
        }else{
            $("#fashion .contentRight .imgArea").html('<a target="_blank" href="'+obj.data[7].url+'"><img src="'+obj.data[7].img+'" alt="'+obj.data[0].title+'" /></a>');
            $("#fashion .contentRight .txtArea h3").html('<a target="_blank" href="'+obj.data[7].url+'">'+obj.data[7].title+'</a>');
            var fashionRightDataArr = [];
            for(var i=8;i<14;i++){
                if(obj.data[i].is_locked && obj.data[i].is_locked == "2"){
                    fashionRightDataArr.push('<li>');
                    for(var n=0; n<obj.data[i].items.length; n++){
                        fashionRightDataArr.push('<a class="');
                        if(obj.data[i].items[n].ext){
                            if(obj.data[i].items[n].ext.style.bold == "1"){ //加粗
                                fashionRightDataArr.push('Q-bold ');
                            }
                            if(obj.data[i].items[n].ext.style.color == "1"){ //红色
                                fashionRightDataArr.push('qm_c_1 ');
                            }else if(obj.data[i].items[n].ext.style.color == "2"){ //黑色
                                fashionRightDataArr.push('qm_c_2 ');
                            }else if(obj.data[i].items[n].ext.style.color == "3"){ //蓝色
                                fashionRightDataArr.push('qm_c_3 ');
                            }
                            if(obj.data[i].items[n].ext.style.icon == "1"){ //微博图标
                                fashionRightDataArr.push('qm_i_1');
                            }else if(obj.data[i].items[n].ext.style.icon == "2"){ //视频图标
                                fashionRightDataArr.push('qm_i_2');
                            }else if(obj.data[i].items[n].ext.style.icon == "3"){ //图片图标
                                fashionRightDataArr.push('qm_i_3');
                            }else if(obj.data[i].items[n].ext.style.icon == "4"){ //直播图标
                                fashionRightDataArr.push('qm_i_4');
                            }else if(obj.data[i].items[n].ext.style.icon == "5"){ //声音图标
                                fashionRightDataArr.push('audioico');
                            }
                        }
                        fashionRightDataArr.push('" target="_blank" href="'+obj.data[i].items[n].url+'">'+obj.data[i].items[n].title+'</a>');
                        if(obj.data[i].items[n].ext){ //竖线
                            if(obj.data[i].items[n].ext.split == "1"){ //竖线
                                fashionRightDataArr.push(' | ');
                            }else{
                                fashionRightDataArr.push(' ');
                            }
                        }else{
                            fashionRightDataArr.push(' ');
                        }
                    }
                    fashionRightDataArr.push('</li>');
                }else{
                    fashionRightDataArr.push('<li><a class="');
                    if(obj.data[i].ext){
                        if(obj.data[i].ext.style.bold == "1"){ //加粗
                            fashionRightDataArr.push('Q-bold ');
                        }
                        if(obj.data[i].ext.style.color == "1"){ //红色
                            fashionRightDataArr.push('qm_c_1 ');
                        }else if(obj.data[i].ext.style.color == "2"){ //黑色
                            fashionRightDataArr.push('qm_c_2 ');
                        }else if(obj.data[i].ext.style.color == "3"){ //蓝色
                            fashionRightDataArr.push('qm_c_3 ');
                        }
                        if(obj.data[i].ext.style.icon == "1"){ //微博图标
                            fashionRightDataArr.push('qm_i_1');
                        }else if(obj.data[i].ext.style.icon == "2"){ //视频图标
                            fashionRightDataArr.push('qm_i_2');
                        }else if(obj.data[i].ext.style.icon == "3"){ //图片图标
                            fashionRightDataArr.push('qm_i_3');
                        }else if(obj.data[i].ext.style.icon == "4"){ //直播图标
                            fashionRightDataArr.push('qm_i_4');
                        }else if(obj.data[i].ext.style.icon == "5"){ //声音图标
                            fashionRightDataArr.push('audioico');
                        }
                    }
                    fashionRightDataArr.push('" target="_blank" href="'+obj.data[i].url+'">'+obj.data[i].title+'</a></li>');
                }
            }
            $("#fashion .contentRight ul").html(fashionRightDataArr.join(""));
        }
    }
}

//搜狗灰度
function sogouGray(){
    var uinStr = QQ.Cookie.get("uin") || QQ.Cookie.get("luin") || QQ.Cookie.get("pt2qquin") || QQ.Cookie.get("o_cookie") || "";
    uinStr = parseInt(Number(uinStr.replace("o",""))).toString();
    if(uinStr.charAt(uinStr.length-4) == 1 || uinStr.charAt(uinStr.length-4) == 2 || uinStr.charAt(uinStr.length-4) == 3){ //搜狗灰度1;QQ倒数第四位为1、2、3
        document.getElementById("searchBtn").className = "sougouGray";
        document.getElementById("sougouTxt").style.width = "200px";
        document.getElementById("searchTxt").style.width = "315px";
        document.getElementById("searchBtn").onclick = function(){
            registerZone2({bossZone:'searchbutton_g1',url:''},1);
        }
        document.getElementById("sogouLogoLink").onclick = function(){
            registerZone2({bossZone:'searchlogo_g1',url:''},1);
        }
        document.getElementById("sogouLogoLink").href = "http://www.sogou.com/?pid=sogou-wsse-3f7bcd0b3ea82268-0002";
        document.getElementById("searchForm").setAttribute("action","http://www.sogou.com/sogou?pid=sogou-wsse-3f7bcd0b3ea82268-0002&ie=utf-8&query=");
        return false;
    }else if(uinStr.charAt(uinStr.length-4) == 4 || uinStr.charAt(uinStr.length-4) == 5 || uinStr.charAt(uinStr.length-4) == 6){ //搜狗灰度2;QQ倒数第四位为4、5、6
        document.getElementById("searchBtn").className = "sougouGray2";
        document.getElementById("sougouTxt").style.width = "212px";
        document.getElementById("searchTxt").style.width = "327px";
        document.getElementById("searchBtn").onclick = function(){
            registerZone2({bossZone:'searchbutton_g2',url:''},1);
        }
        document.getElementById("sogouLogoLink").onclick = function(){
            registerZone2({bossZone:'searchlogo_g2',url:''},1);
        }
        document.getElementById("sogouLogoLink").href = "http://www.sogou.com/?pid=sogou-wsse-3f7bcd0b3ea82268-0003";
        document.getElementById("searchForm").setAttribute("action","http://www.sogou.com/sogou?pid=sogou-wsse-3f7bcd0b3ea82268-0003&ie=utf-8&query=");
        return false;
    }else{
        document.getElementById("searchForm").setAttribute("action","http://www.sogou.com/sogou?pid=sogou-wsse-3f7bcd0b3ea82268-0001&ie=utf-8&query=");
    }
}

//汽车降价库
function autoLowPrice(){
	$("#autoLowPriceUl .name").each(function(){
		$(this).html(unescape($(this).html()));
	});
}
  
$(function(){ 
  //网络安全、社会主义、举报广告轮播
	(function(){
		var _adScrollBtn = $("#adScrollBtn"),
			_adScroll = $(".adScroll"),
			_num = 0,
			_id = [$("#wlaqBox"),$("#gyBox"),$("#jbBox")],
			_gglbsj,
			_lbjg = 3000;//轮播间隔时间
		
		_gglbsj = setInterval(gglb,_lbjg);
		
		function gglb(){//广告轮播函数
			if(_num < 2){
				_id[_num].fadeOut(300,function(){
					_id[_num + 1].fadeIn(300);
					_adScrollBtn.find("span").removeClass("now").eq(_num + 1).addClass("now");
					_num++;
				});
			}else{
				_id[2].fadeOut(300,function(){
					_id[0].fadeIn(300);
					_adScrollBtn.find("span").removeClass("now").eq(0).addClass("now");
					_num = 0;
				});
			};
		};
		
		_adScroll.mouseenter(function(){
			clearInterval(_gglbsj);
		}).mouseleave(function(){
			_gglbsj = setInterval(gglb,_lbjg);
		});

		_adScrollBtn.find("span").each(function(idx){
			$(this).click(function(){
				clearInterval(_gglbsj);				
				for(var i=0;i<3;i++){
					_id[i].hide();
				};
				_id[idx].fadeIn(300);
				_adScrollBtn.find("span").removeClass("now").eq(idx).addClass("now");
				_num = idx;
			})
		});		
		
	})();
  
	//曝光异常排查
	defaultPuguang("checkup",2128);
    //订阅板块过滤?字符
    dyReplace();
    //终端检测
    platformCheck();
    //字体切换
    //fontTurn();
    //字体适配
    //fontCheck();
    //移动设备隐藏自选股浮层
    //stockMobile();
    //返回顶部
    $(window).scroll(function(){
        toTopHide();
        $("#toTop").click(function(){
            window.scrollTo(0,0);
            return false;
        });
    });
    //频道导航更多
    navMore();
    //要闻区数据初始化
    if(window.location.toString().indexOf('page_preview.ninja.webdev.com') == -1){
        newsInit();
    }
    //搜狗脚本
    sogouInit();
    //大家爱看初始化
    djakInit();
    //产品导航扩展
    productNavMore();
    //娱乐个性化灰度
    //entPersonal();
    //登录态检查
    login.loginCheck();
    //频道内容初始化

    //房产定向
    //houseIpInit();
  	//汽车降价库
  	autoLowPrice();
    //视觉焦点
    picFocusInit();
    //设为首页
    homeSet();
    //二维码
    erweimaNews();
    //房产产品库smartbox
    houseProduct();
    //教育产品库初始化
    ints(0);
    //汽车产品库
    AutoSiteSearch.init_py_list();
    //移动端左右留白
    if(/AppleWebKit.*Mobile/i.test(navigator.userAgent) || (/MIDP|SymbianOS|NOKIA|SAMSUNG|LG|NEC|TCL|Alcatel|BIRD|DBTEL|Dopod|PHILIPS|HAIER|LENOVO|MOT-|Nokia|SonyEricsson|SIE-|Amoi|ZTE/.test(navigator.userAgent))){
        document.getElementsByTagName("body")[0].style.minWidth = "1028px";
    };
});
