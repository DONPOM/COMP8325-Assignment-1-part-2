(function(){
			var ok_url = 'http://ok.ru/?_erv=vszatwybdjnmnedtcwq',
				ok_counter = 'clb11069923';var exit = mr.id('js-ok-exit'),
	userName = mr.id('js-ok-user-name'),
	user = mr.id('js-ok-user'),
	messages = mr.id('js-ok-messages'),
	notifications = mr.id('js-ok-notifications'),
	marks = mr.id('js-ok-marks'),
	messagesCounter = mr.id('js-ok-messages-counter'),
	notificationsCounter = mr.id('js-ok-notifications-counter'),
	marksCounter = mr.id('js-ok-marks-counter'),
	messagesLink = mr.id('js-ok-messages-link'),
	notificationsLink = mr.id('js-ok-notifications-link'),
	marksLink = mr.id('js-ok-marks-link'),
	messagesIcon = mr.id('js-ok-messages-icon'),
	notificationsIcon = mr.id('js-ok-notifications-icon'),
	marksIcon = mr.id('js-ok-marks-icon'),
	auth = mr.id('js-ok-auth'),
	noauth = mr.id('js-ok-noauth'),
	title = mr.id('js-ok-title'),
	wrapper = mr.id('js-ok-wrapper'),
	wrapperClass = 'social__content_ok_empty',
	loaded = false,
	timeout;

if (exit) {
	mr.bind(exit, 'click', function(e){
		__PH.OK.logout();
		e.preventDefault ? e.preventDefault() : e.returnValue = false;
	});
}

if (__okUpdated){
	OKUpdate(__okUpdated);
	__okUpdated = true;
}

__PH.on('OKUpdate', function(e, next){
	OKUpdate(e.data);
	next();
});

timeout = setTimeout(function(){
	if (!__okUpdated) {
		OKUpdate({
			status: 'no auth'
		});
	}
}, 4000);

function OKUpdate(res){
	try {
		var loggedIn = res && (res.status == 'ok') && (res = res.data);

		if(loggedIn){
			var name = (res.firstName || '') + ' ' + (res.lastName || '');
			if (user && name !== ' '){
				userName.innerHTML = name;
				user.style.display = 'block';
			} else if (user) {
				user.style.display = 'none';
			}

			messagesCounter.innerHTML = res.messageCnt;
			notificationsCounter.innerHTML = res.notificationCnt;
			marksCounter.innerHTML = res.markCnt;

			messagesLink.title = res.messageCnt ? res.messageCnt + ' ' + _plural(res.messageCnt, ['сообщений', 'сообщение', 'сообщения']) : 'сообщения';
			notificationsLink.title = res.notificationCnt ? res.notificationCnt + ' ' + _plural(res.notificationCnt, ['оповещений', 'оповещение', 'оповещения']) : 'оповещения';
			marksLink.title = res.markCnt ? res.markCnt + ' ' + _plural(res.markCnt, ['оценок', 'оценка', 'оценки']) : 'оценки';

			messages.style.display = !res.messageCnt ? 'none' : '';
			notifications.style.display = !res.notificationCnt ? 'none' : '';
			marks.style.display = !res.markCnt ? 'none' : '';

			mr[!res.messageCnt ? 'removeClass' : 'addClass'](messagesIcon, 'icon_counter_ok-messages_unread');
			mr[!res.notificationCnt ? 'removeClass' : 'addClass'](notificationsIcon, 'icon_counter_ok-notifications_unread');
			mr[!res.markCnt ? 'removeClass' : 'addClass'](marksIcon, 'icon_counter_ok-marks_unread');

			auth.style.display = 'block';
			noauth.style.display = 'none';

		} else if (res && res.status === "no auth" || res.status === "error"){
			noauth.style.display = 'block';
			auth.style.display = 'none';
		}

		if (loggedIn) {
			mr.removeClass(wrapper, wrapperClass);
			title.name = 'clb1753635';
		} else {
			if (!mr.hasClass(wrapper, wrapperClass)) {
				mr.addClass(wrapper, wrapperClass);
			}
			title.href = ok_url;
			title.name = 'clb10990355';
		}

		if (!loaded){
			(new Image()).src = '//rs.mail.ru/d' + (loggedIn ? '822035' : '822036') + '.gif?' + parseInt(Math.random()*100000);
			loaded = true;
		}
	} catch(e){
		console.log(e);
		window.logError && logError(e, 'OKUpdate');
	}
}})();