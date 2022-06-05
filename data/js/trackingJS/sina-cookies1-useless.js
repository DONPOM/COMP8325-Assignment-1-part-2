
        ;(function() {
		//var shmIPlookup = null;
            var handle = function(info, city) {
                // info为remote_ip_info,city为与cookie中的city比较后的城市，cookie优先
                var province = info.province;
                city = info.city;
		//var province = shmIPlookup.getPNameByCName(city);
		//city = city;

                var desc = info.desc;
                if (info.ret != 1) {
                    return;
                }

                var autoJS = '';
                var areaJS = '';
                var mtJS = '';
                switch (province) {
                   case '上海':
                        areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/sh.js';
                        autoJS = 'http://auto.sina.com.cn/867/2012/1217/14_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/shanghai.js';
                        break;

                    case '四川':
                        areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/sc.js';
                        autoJS = 'http://auto.sina.com.cn/867/2012/1217/12_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/sichuan.js';
                        break;

                    case '河南':
                        areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/henan.js';
                        autoJS = 'http://auto.sina.com.cn/867/2012/1217/1_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/henan.js';
                        break;

                    case '广东':
                        areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/gd.js';
                        autoJS = 'http://auto.sina.com.cn/867/2013/1031/26.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/guangdong.js';
                        break;

                    case '福建':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/fj.js';
						autoJS = 'http://auto.sina.com.cn/867/2012/1217/7_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/fujian.js';
                        if (city == '厦门' || desc == 'Xiamen' || city == '漳州' || desc == 'Zhangzhou' || city == '泉州' || desc == 'Quanzhou') 
						{ areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/fj_mn.js'; }
                        break;

                    case '浙江':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/zj.js';
						autoJS = 'http://auto.sina.com.cn/867/2012/1217/4_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/zhejiang.js';
                        if (city == '宁波' || desc == 'Ningbo') 
						{ areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/zj_nb.js'; }
                        break;

                    case '重庆':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/cq.js';
						autoJS = 'http://auto.sina.com.cn/867/2012/1217/11_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/chongqing.js';
                        break;

                    case '湖南':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/hunan.js';
						autoJS = 'http://auto.sina.com.cn/867/2012/1217/10_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/hunan.js';
                        break;

                    case '湖北':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/hb.js';
						autoJS = 'http://auto.sina.com.cn/867/2012/1217/9_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/hubei.js';
                        break;

                    case '陕西':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/sx.js';
						autoJS = 'http://auto.sina.com.cn/867/2012/1217/13_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/shannxi.js';
                        break;

                    case '辽宁':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/ln.js';
						autoJS = 'http://auto.sina.com.cn/867/2012/1217/3_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/liaoning.js';
                        if (city == '大连' || desc == 'Dalian') 
						{ areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/ln_dl.js'; }
                        break;

                    case '黑龙江':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/hlj.js';
						autoJS = 'http://auto.sina.com.cn/867/2012/1217/5_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/heilongjiang.js';
                        break;

                    case '安徽':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/ah.js';
						autoJS = 'http://auto.sina.com.cn/867/2012/1217/8_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/anhui.js';
                        break;

                    case '河北':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/hebei.js';
						autoJS = 'http://auto.sina.com.cn/867/2012/1217/6_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/hebei.js';
                        break;

                    case '江苏':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/jiangsu.js';
						autoJS = 'http://auto.sina.com.cn/867/2012/1217/2_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/jiangsu.js';
                        if (city == '无锡') { areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/jiangsu_wx.js'; }
                        if (city == '苏州') { areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/sz.js'; }
                        break;

                    case '天津':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/tj.js';
						autoJS = 'http://auto.sina.com.cn/867/2013/0105/15_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/tianjin.js';
                        break;

                    case '山西':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/shanxi.js';
						autoJS = 'http://auto.sina.com.cn/867/2013/0105/16_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/shanxi.js';
                        break;

                    case '吉林':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/jl.js';
						autoJS = 'http://auto.sina.com.cn/867/2013/0105/19_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/jilin.js';
                        break;

                    case '江西':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/jx.js';
						autoJS = 'http://auto.sina.com.cn/867/2013/0105/17_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/jiangxi.js';
                        break;

                    case '海南':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/hainan.js';
						autoJS = 'http://auto.sina.com.cn/867/2013/0105/18_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/hainan.js';
                        break;

                    case '山东':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/sd.js';
						autoJS = 'http://auto.sina.com.cn/867/2013/0105/20_1.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/shandong.js';
                        if (city == '青岛') { areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/qd.js'; }
                        break;

                    case '广西':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/gx.js';
						autoJS = 'http://auto.sina.com.cn/867/2013/0528/22.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/guangxi.js';
                        break;

                    case '云南':
						autoJS = 'http://auto.sina.com.cn/867/2013/0528/23.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/yunnan.js';
                        break;

                    case '贵州':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/gz.js';
						autoJS = 'http://auto.sina.com.cn/867/2013/0528/24.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/guizhou.js';
                        break;

                    case '甘肃':
						autoJS = 'http://auto.sina.com.cn/867/2013/0528/25.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/gansu.js';
                        break;

                    case '新疆':
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/xinjiang.js';
                        break;

                    case '内蒙古':
						areaJS = 'http://www.sina.com.cn/js/67/sinaindex/2013/nmg.js';
						autoJS = 'http://auto.sina.com.cn/867/2013/1126/27.js';
						mtJS	= 'http://www.sina.com.cn/js/135/2013/ipmt/neimenggu.js';
                        break;

                    default:
                        //autoJS = '';
                        //areaJS = '';
                        break;
                }
                // 汽车ip定向
                if(autoJS){
                    jsLoader({
                      url : autoJS
                    });
                }
                //地方站ip定向
                if(areaJS){
                    jsLoader({
                      url : areaJS
                    });
                }
                //媒拓ip定向
                if(mtJS){
                    jsLoader({
                      url : mtJS
                    });
                }
            };

            jsLoader({
                name: 'shm',
                callback: function() {
                    SHM.home.iplookup.load(function(info, city) {
                        handle(info, city);
                    });
                }
            });

            //MT ip定向
/*
            jsLoader({
                name: 'ipMT',
                url:'http://www.sina.com.cn/js/67/index2013/ipMT.js'
            });
*/

        })();

//房产ip定向
;(function(){
	var API = 'http://ip.house.sina.com.cn/sina_sanshou_2010.php';
	var render = function() {
		for (var i = 0, len = SI_IP_House_.length; i < len; i++) {
			var item = SI_IP_House_[i];
			var node = document.getElementById('SI_IP_House_'+i);
			if (item&&node) {
				node.innerHTML = item;
			}
		}
	};
	jsLoader({
	    name: 'ipHouse',
	    url:API,
	    callback: function() {
	    	render();
	    }
	});
})();
