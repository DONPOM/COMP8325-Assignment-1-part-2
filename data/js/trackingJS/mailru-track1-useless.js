function Placeholder(i, p){
	if ('placeholder' in i){
		i.placeholder = p.innerHTML;
		p.style.display = 'none';

		this.native = true;

		return this;
	}

	this._hasFocus = false;
	this.i = i;
	this.p = p;
	var _this = this;
	mr.bind(i, 'focus', function(){_this.focus()});
	mr.bind(i, 'blur', function(){_this.blur()});
	this.toggle();

	function t(){
		if (!_this._hasFocus){
			_this.toggle();
		}
		setTimeout(t, 50);
	}
	setTimeout(t, 50);
}

Placeholder.prototype = {
	focus: function(){
		if (this.native){
			return;
		}
		this._hasFocus = true;
		this.toggle();
	},
	blur: function(){
		if (this.native){
			return;
		}
		this._hasFocus = false;
		this.toggle();
	},
	toggle: function(){
		if (this.native){
			return;
		}
		this.p.style.display = this.i.value || this._hasFocus ? 'none' : 'block';
	}
};

function rotateSlots(){}
mr.onFocusBannerReload =true;mr.domains =["mail.ru","inbox.ru","list.ru","bk.ru"];
		mr.blockVersions ={"newsVersion":3,"mapsVersion":3,"trafficVersion":3,"quotationsVersion":4,"posterVersion":4,"scheduleVersion":2,"horoscopeVersion":6,"gamesVersion":6,"textBannerVersion":3,"socialOkVersion":3,"galleryVersion":2,"liveVersion":1,"version_event":1,"version_currency":1};

		mr.externalDomainsEnabled =false;

		mr.logViewport =false;
		mr.slidotype_id = '9975547';
		mr.ajaxUrl = 'https:\/\/mail.ru?json=1&nano=1';