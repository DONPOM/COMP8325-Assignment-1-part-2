/**
 * 腾讯首页搜索热词推荐 
 * @author  HuangBingcheng
 * @charset UTF-8
 * @version 1.0.2
 * 
 * 此次修改了热词框的显示逻辑，只在搜索类型为网页时显示
 * 并且在鼠标移入左侧搜索类型选择框时不显示
 * 移出类型选择框时，类型为网页，并且搜索框无文字时显示热词
 * 
 * 用法
 * initSogouHotWord(document.forms['soso_search_box'], document.getElementById('sougouTxt'));
 */
(function() {
    var hotWordUrl = 'http://wap.sogou.com/data/hotwords/qq_index.js',
        pid = 'sogou-wsse-8baca01b732cf56f',
        queryUrl = 'http://www.sogou.com/tx?hdq=' + pid + '&query=',
        suggestionUrl = 'http://www.sogou.com/tx?hdq=sogou-wsse-3f7bcd0b3ea82268&ie=utf-8&query=',

        data,
        maxHotWord, // 最多展示热词条数
        // 用来存热词DOM
        searchForm, searchBox, suggestionDiv, suginnerDiv, sugTitleP, sugListUl,
        // 搜索框搜索建议
        searchSmart = document.getElementById('searchSmart'),
        // 搜索类型
        searchType = document.getElementById('searchSelected'),
        searchTab = document.getElementById('searchTab'),
        timer1 = null,
        timer2 = null,
        isShow = false,
        sugTitle = '今日热词',

        escKeyCode = 27, // Esc
        upKeyCode = 38,  // 方向键上
        downKeyCode = 40,// 方向键下
        curHotWordIndex = -1, //当前hover热词序号
        hotWordArr = [];

    /**
     * 入口函数
     * @param  {DOM} form    搜索表单元素，热词html会插入到此标签内
     * @param  {DOM} sBox    搜索框input
     * @param  {int} maxWord 【可选，默认5条】热词最大展示条数
     */
    function init(form, sBox, maxWord) {
        maxHotWord = maxWord || 5;
        searchForm = form;
        searchBox = sBox;

        addEvent(searchBox, 'click', function() {
            if (searchBox.value.length > 0) return;

            getHotWords();
        });

        addEvent(searchBox, 'keydown', function(ev) {
            var e = ev || window.event,
                keyCode = e.keyCode;

            if (!isShow) {
                return;
            }

            if (keyCode == upKeyCode) {
                curHotWordIndex--;
                if (curHotWordIndex < 0) {
                    curHotWordIndex = maxHotWord - 1;
                }

                var curLi = sugListUl.getElementsByTagName('li')[curHotWordIndex];

                removeHoverClass(sugListUl.getElementsByTagName('li'));
                curLi.className = 'over';
                searchBox.value = curLi.innerHTML;
                return false;
            }

            if (keyCode == downKeyCode) {
                curHotWordIndex++;
                if (curHotWordIndex > maxHotWord - 1) {
                    curHotWordIndex = 0;
                }

                var curLi = sugListUl.getElementsByTagName('li')[curHotWordIndex];

                removeHoverClass(sugListUl.getElementsByTagName('li'));
                curLi.className = 'over';
                searchBox.value = curLi.innerHTML;
                return false;
            }

            // 修复在搜狗浏览器高速模式下，用中文输入法输入文字后按回车键无法隐藏热词框BUG
            timer1 = setTimeout(function() {
                clearTimeout(timer1);
                if (searchBox.value.length > 0) {
                    removeHotWord();
                }
            }, 100);
        });

        addEvent(searchBox, 'keypress', function(ev) {
            var e = ev || window.event,
                keyCode = e.keyCode;

            if (!isShow) {
                return;
            }

            if (keyCode == upKeyCode || keyCode == downKeyCode) {
                return false;
            }
        });

        addEvent(searchBox, 'keyup', function(ev) {
            var e = ev || window.event,
                keyCode = e.keyCode;

            if (keyCode == upKeyCode || keyCode == downKeyCode) {
                return false;
            }

            if (searchBox.value.length > 0 || keyCode == escKeyCode) {
                removeHotWord();                
            } else {
                getHotWords();
            }
        });
    }

    /*
     * 获取热词
     */
    function getHotWords() {
        jsonp(hotWordUrl);
    }

    /**
     * 最多展示5条热词
     * @param  {arr} data jsonp返回的数据
     */
    window.hotwords = function(_data) {
        if (_data.length > maxHotWord) {
            data = _data.slice(0, maxHotWord);
        } else {
            data = _data;
        }

        drawHtml(searchForm);
    };

    /**
     * 生成html
     * @param  {DOM} parent 父级，热词html会插入到此标签内
     */
    function drawHtml(parent) {
        if(data.length < 1 || isShow || searchType.innerHTML != '网页') return;

        suggestionDiv = document.createElement('div'),
        suginnerDiv = document.createElement('div'),
        sugTitleP = document.createElement('p'),
        sugListUl = document.createElement('ul');

        suggestionDiv.className = 'suggestion';
        suginnerDiv.className = 'suginner';
        sugTitleP.className = 's_title';
        sugListUl.className = 'suglist';

        sugTitleP.innerHTML = sugTitle;

        for (var i = 0, len = data.length; i < len; i++) {
            var li = document.createElement('li');

            li.innerHTML = data[i].word;
            sugListUl.appendChild(li);

            hotWordArr.push(data[i].word);

            // 超出热词最大展现条数
            if (i >= maxHotWord) { break; }
        }

        suginnerDiv.appendChild(sugTitleP);
        suginnerDiv.appendChild(sugListUl);
        suggestionDiv.appendChild(suginnerDiv);

        parent.appendChild(suggestionDiv);

        searchForm.action = queryUrl;
        if (searchSmart) {searchSmart.style.visibility = 'hidden';}

        bind();
        isShow = true;
    }

    function bind() {
        var li = sugListUl.getElementsByTagName('li');

        // li add hover class
        addEvent(sugListUl, 'mouseover', function(ev) {
            var e = window.event || ev,
                srcEle = e.srcElement || e.target;

            if (srcEle.tagName.toUpperCase() == 'LI') {
                for (var i = 0, len = li.length; i < len; i++) {
                    li[i].className ='';
                }

                srcEle.className = 'over';

                curHotWordIndex = hotWordArr.indexOf(srcEle.innerHTML);
            }

            customStopPropagation(e);
        });

        // li remove hover class
        addEvent(sugListUl, 'mouseout', function(ev) {
            var e = window.event || ev,
                srcEle = e.srcElement || e.target;

            removeHoverClass(li);

            curHotWordIndex = hotWordArr.indexOf(srcEle.innerHTML);
            customStopPropagation(e);
        });

        // click hot word open sogou result page
        addEvent(sugListUl, 'click', function(ev) {
            var e = window.event || ev,
                srcEle = e.srcElement || e.target,
                queryWord, url;

            if (srcEle.tagName.toUpperCase() == 'LI') {
                queryWord = srcEle.innerHTML;
                searchBox.value = queryWord;
                url = queryUrl + encodeURI(queryWord);

                window.open(url);
            }

            removeHotWord();

            customStopPropagation(e);
        });

        addEvent(document.body, 'click', function() {
            removeHotWord();
        });

        addEvent(searchType, 'mouseover', function() {
            if (suggestionDiv) {
                removeHotWord();                    
            }
        });

        addEvent(searchType, 'mouseout', function(ev) {
            var e = ev || e.event,
                toEle = e.toElement || e.relatedTarget,
                toEleParent = isParent('DIV', toEle);

            if (toEleParent == searchTab) {
                return;
            }
            if (!suggestionDiv && searchBox.value.length == 0 && document.activeElement.id == 'sougouTxt') {
                getHotWords();
            }
        });

        addEvent(searchTab, 'mouseover', function(ev) {
            var e = ev || window.event,
                srcEle = e.srcElement || e.target;

            if (isParent('DIV', srcEle) == searchTab) {
                if (suggestionDiv) {
                    removeHotWord();
                }
            }
        });

        addEvent(searchTab, 'mouseout', function(ev) {
            var e = ev || window.event,
                srcEle = e.srcElement || e.target,
                toEle = e.toElement || e.relatedTarget;

            if (isParent('DIV', toEle) == searchTab) {
                return;
            }
            
            if (!suggestionDiv && searchBox.value.length == 0 && document.activeElement.id == 'sougouTxt') {
                getHotWords();
            }
        });
    }

    function removeHoverClass(nodeList) {
        if(!nodeList) return;

        for (var i = 0, len = nodeList.length; i < len; i++) {
            nodeList[i].className = '';
        }
    }

    /**
     * 删除热词推荐框
     */
    function removeHotWord() {
        if (isShow) {
            searchForm.action = suggestionUrl;
            searchForm.removeChild(suggestionDiv);
            suggestionDiv = null;
            curHotWordIndex = -1;
            isShow = false;
        }
    }

    /**
     * jsonp，时间戳字段名为t
     * @param  {string} url 接口地址
     * @return 无
     */
    function jsonp(url) {
        if (!url) {
            return;
        }

        if (url.indexOf('?') < 0) {
            url += ('?t=' + new Date().getTime());
        } else {
            url += ('&t=' + new Date().getTime());
        }

        var script = document.createElement('script');

        script.src = url;
        script.setAttribute("type","text/javascript");
        script.async = true;

        document.body.appendChild(script);

        if (script.readyState == 4) {
            script.onreadystatechange = function() {
                this.onload = null;
                this.parentNode.removeChild(this);
            };
        } else {
            script.onload =function(){
                this.onload = null;
                this.parentNode.removeChild(this);
            };
        }
    }

    /**
     * 绑定事件
     * @param {DOM}      obj  绑定对象
     * @param {string}   type 事件名，不带on
     * @param {Function} fn   事件函数
     */
    function addEvent(obj, type, fn) {
        obj.attachEvent ? obj.attachEvent('on'+type, fn) : obj.addEventListener(type, fn, false);
    }

    /**
     * 阻止冒泡
     * @param  {event} e 
     */
    function customStopPropagation(e) {
        if (e && e.stopPropagation) {
            e.stopPropagation();
        } else {
            e.cancelBubble = true;
        }
    }

    /**
     * 判断是否是父级
     * @param  {[type]}  sParentTagName [description]
     * @param  {[type]}  obj            [description]
     * @return {Boolean}                [description]
     */
    function isParent(sParentTagName, obj){
        while(obj && obj.tagName != 'BODY'){
            if(obj.tagName.toUpperCase() == sParentTagName) return obj;

            obj = obj.parentNode;
        }
        return false;
    }

    window.initSogouHotWord = init;
})();/*  |xGv00|08261400eb4fe53ac0074fd2948aa458 */