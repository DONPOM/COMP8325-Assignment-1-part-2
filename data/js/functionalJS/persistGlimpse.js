define([
	'spine',
	'internal/sitebuilder/common/log',
	'internal/sitebuilder/builderChrome/nodes',
	'translate!webs.util',
	'internal/sitebuilder/common/eventBuffer',
	webs.props.sitebuilderApiUrl + '/primus/primus.js',
	'internal/sitebuilder/common/jquery.loadingSpinner'
], function define_persist(Spine, log, nodes, translate, eventBuffer, io) {
	/* jshint ignore:start */
	var saveQueue = [],
		saving = false,
		nodeErrors = {},
		totalErrors = 0,
		catastrophicErrors = 0,
		lastSaveAttempt = 0,
		lastSaveSuccess = 0,
		lastSaveError = 0,
		primus,
		primusAuthFailed = false,
		primusState = 0,
		guidMap = {},

		messageHandlers = {
			'reauthenticate': function reauthenticate(){
				$.ajax({
					url: "/api/sites/" + webs.site.id + "/glimpse/token",
					cache: false,
					success: function(data) {
						data.msg = 'authenticate';
						primus.write(data);
					},
					error: function(data) {
						// Don't even try again...
						primusAuthFailed = true;
						primus.end();
						catastrophicError();
					}
				});
			},
			'authFailure': function authFailure(){
				primusAuthFailed = true;
				catastrophicError();
			},
			'identifyModule': function identifyModule(data){
				// Add the module's ID if need be.
				if (guidMap[data.guid]) {
					guidMap[data.guid].setId(data.id);
					handleSuccess(guidMap[data.guid], 'Saved ' + guidMap[data.guid].moduleType + ' module. ' + guidMap[data.guid].id);
					delete guidMap[data.guid];
				}
			}
		},

		connectPrimus = function(onConnectCallback) {
			if(primus) {
				// Clean out listeners to prevent memory leaks
				primus.removeListener('end');
				primus.removeListener('open');
				primus.removeListener('error');
				primus.removeListener('data');
			}

			// (re)connect
			primus = Primus.connect(webs.props.sitebuilderApiUrl);
			log.debug('glimpse connecting', arguments);
			primusState = -1;

			primus.on('open', function primusOpen(){
				primusState = 1;
				log.debug('glimpse open');
				if(typeof(onConnectCallback) === 'function') onConnectCallback();
			});

			primus.on('error', function primusError(){
				log.warn('glimpse error', arguments);
			});

			// Keep state (whether we're connected)
			primus.on('end', function primusDisconnected(){
				log.debug('glimpse end', arguments);
				primusState = 0;
			});

			// Listen for stuff
			primus.on('data', function incomingPrimusMessage(data){
				var msg = data.msg;
				if(messageHandlers[msg]) {
					messageHandlers[msg](data);
				} else {
					log.trigger('persist', 'warn', 'Unknown glimpse message', data);
				}
			});
			return primus;
		},

		catastrophicError = function catastrophe() {
			// Catastrophic failure.. Either the server is down or the network is down or something.
			// Remove the onbeforeunload because it's not going to help anything.
			window.onbeforeunload = function persist_onbeforeunload_stub(e){ return undefined; };
			// Let the user know what's going on.
			alert(translate('webs.util.error.UNABLE_TO_SAVE'));
			// Track that this happened.
			eventBuffer.trigger("save:failure", {
//				count: numErrors,
				catastrophicErrors: catastrophicErrors,
				totalErrors: totalErrors,
				lastSaveAttempt: (+new Date()) - lastSaveAttempt,
				lastSaveSuccess: (+new Date()) - lastSaveSuccess,
				lastSaveError:   (+new Date()) - lastSaveError
			});
		};


		_save = function persist__save() {
			if(saveQueue.length === 0) {
				saving = false;
				return false;
			}

			if(primusAuthFailed) {
				log.warn('auth failed. not saving.');
				return;
			}

			if(primusState === 0) {
				connectPrimus(_save);
				return;
			} else if(primusState == -1) {
				// Chill out.. we're connecting.
				return;
			}

			saving = true;

			var curr = saveQueue.shift(),
				tree = curr.tree();

			if(!tree && !curr.deleteFromTreeId){
				log.trigger("persist", "error", "Trying to save item without tree", curr, tree);
				return setTimeout(_save, 0);
			}

			if(curr.type === 'module') {
				if(curr.id) { // existing module
					for(var i=0,len=saveQueue.length;i<len;i++) {
						// If the module exists later in the queue, skip it this time and it'll get picked up later.
						if(saveQueue[i].type === 'module' && saveQueue[i].id && saveQueue[i].id === curr.id) {
							return setTimeout(_save, 0);
						}
					}
				}
				if (curr.deleteFromTreeId){
					lastSaveAttempt = +new Date();

					// Send a delete request to Glimpse.
					primus.write({msg: "deleteModule", tree: curr.deleteFromTreeId, id: curr.id});

					lastSaveSuccess = +new Date();
					handleSuccess(curr, "Deleted " + curr.moduleType + " module. " + curr.id);

				} else {
					lastSaveAttempt = +new Date();

					// Create a guid to reference later if this is a new module.
					var guid;

					// If this module is new, save it for later identification.
					if (!curr.id) {
						guid = new Date().getTime();
						guidMap[guid] = curr;
					}

					// Emit a save request on the socket to Glimpse.
					primus.write({
						msg: "upsertModule",
						guid: guid,
						module: curr.toModuleJSON(),
						tree: tree.id,
						id: curr.id
					});

					lastSaveSuccess = +new Date();

					// Call it saved.
					// TODO: Fix this to make it work for realz.
					if (curr.id) {
						handleSuccess(curr, 'Saved ' + curr.moduleType + ' module. ' + curr.id);
						Spine.Events.trigger("module:save",curr);
					}


				}
			} else if(curr.type === 'tree') {
				if(curr.isDirty()) {
					tree.dirty(false);
					var putData = curr.toJSON();
					lastSaveAttempt = +new Date();

					// Send a save request for the tree to Glimpse.
					primus.write({msg: 'saveTree', id: curr.id, data: putData});

					lastSaveSuccess = +new Date();
					handleSuccess(curr,'Saved Tree ' + curr.id + '.');
					Spine.Events.trigger("tree:save",curr);

				} else {
					// Moving right along...
					log.trigger("persist", "info", "No need to save tree that isn't dirty.");
					return setTimeout(_save, 0);
				}
			} else {
				eventBuffer.trigger("save:failure:strangeItem");
				log.trigger("persist", "error", "Unable to save an item that is neither module nor tree.");
				return setTimeout(_save, 0);
			}

			return true;
		},
		handleSuccess = function persist_handleSuccess(node, msg){
			log.trigger("persist", "info", msg, node);
			if(typeof(nodeErrors[node]) !== 'undefined'){
				nodeErrors[node] = 0;
			}
			setTimeout(_save, 0);
		},
		handleError = function persist_handleError(node, msg, beforeSave){
			++totalErrors;
			log.trigger("persist", "error", msg, node);
			var numErrors = nodeErrors[node];
			if(typeof(numErrors) === 'undefined'){
				nodeErrors[node] = numErrors = 0;
			}

			if(numErrors <= 2){
				if(typeof(beforeSave) === 'function'){
					beforeSave.call();
				}
				setTimeout(_save, (numErrors + 1) * (numErrors + 1) * 5000);
			} else {
				++catastrophicErrors;
				// Catastrophic failure.. Either the server is down or the network is down or something.
				// Remove the onbeforeunload because it's not going to help anything.
				window.onbeforeunload = function persist_onbeforeunload_stub(e){ return undefined; };
				// Let the user know what's going on.
				alert(translate('webs.util.error.UNABLE_TO_SAVE'));
				// Track that this happened.
				eventBuffer.trigger("save:failure", {
					count: numErrors,
					catastrophicErrors: catastrophicErrors,
					totalErrors: totalErrors,
					lastSaveAttempt: (+new Date()) - lastSaveAttempt,
					lastSaveSuccess: (+new Date()) - lastSaveSuccess,
					lastSaveError:   (+new Date()) - lastSaveError
				});
			}
			nodeErrors[node]++;
		},
		/**
		 * Push a module to the queue to be saved
		 */
		push = function persist_push(model) {
			saveQueue.push(model);
			log.trigger("persist", "info", "Pushed an item onto the save queue: " + saveQueue.length + " items.", model);
		},

		/**
		 * Dump the saveQueue to the back-end
		 */
		save = function persist_save() {
			if(saving) return;

			// push the current active module to the queue (otherwise, the active node would never get saved!)
			var currentController = bldr.pageController.getCurrentController();
			if(currentController) {
				currentController.saveModule();
			}

			// push all dirty trees to the queue
			var trees = require('internal/sitebuilder/builderChrome/nodes').Tree.instances;
			for(var i in trees) {
				if(trees.hasOwnProperty(i)) {
					tree = trees[i];
					if(tree.isDirty()) saveQueue.push(tree);
				}
			}
			setTimeout(_save, 0);
		},

		/**
		 * Grab the data for a tree
		 */
		load = function persist_load(id, successCallback, errorCallback) {
			if (typeof errorCallback != "function") {
				errorCallback = function errorCallback_stub(){
					log.trigger("persist", "error", "persist_load called without errorCallback.");
				};
			}
			$.ajax({
				url: '/s/sitebuilder/api/trees/' + id,
				cache: false,
				success: function persist_load_success(data, textStatus, jqXHR) {
					log.trigger("Trees", "info", "Loading Tree: have data for tree: " + id);
					if(!(data instanceof Object)) return errorCallback();
					return successCallback(data);
				},
				error: errorCallback
			});
		},

		pub = {
			push: push,
			save: save,
			load: load
		};
	/* jshint ignore:end */

	// Prompt user when leaving the page with unsaved changes.
	var unsavedChangesWarning = "You have unsaved changes. Please leave the page open while we save them.",
		unsavedChanges = function persist_unsavedChanges(){
			return (saveQueue.length > 0);
		},
		unsavedChangesMessageViewedCount = 0;

	var intendedLinkTarget = null;

	$(function(){
		top.$("#header a").live("mousedown", function(e){
			var $link = $(e.currentTarget);
			if ($link.attr("target") == "_blank" || $link.attr("href") == "#" || $link.hasClass("w-page-options-link")) {
				intendedLinkTarget = null;
			} else {
				intendedLinkTarget = $link.attr("href");
			}
		});
	});

	// Preventing unload is a tricky business. Test carefully when changing anything below.
	window.onbeforeunload = function persist_onbeforeunload(e){
		var wasSaving = saving;
		if(!unsavedChanges()) return undefined;

		// kick off save immediately
		unsavedChangesMessageViewedCount++;
		save();
		eventBuffer.trigger("unsavedChanges", {
			count: unsavedChangesMessageViewedCount,
			queueLength: saveQueue.length,
			wasSaving: wasSaving,
			catastrophicErrors: catastrophicErrors,
			totalErrors: totalErrors,
			lastSaveAttempt: (+new Date()) - lastSaveAttempt,
			lastSaveSuccess: (+new Date()) - lastSaveSuccess,
			lastSaveError:   (+new Date()) - lastSaveError
		});
		log.trigger("persist", "error", "User attempting to change window with changes unsaved.");

		if(!e) e = window.event;

		// IE
		e.cancelBubble = true;
		e.returnValue = unsavedChangesWarning;

		// FF
		if (e.stopPropagation) {
			e.stopPropagation();
			e.preventDefault();
		}

		setTimeout(function placeCover(){
			var $saveCover = top.$('<div></div>')
				.addClass('w-persist-flush-cover')
				.append(top.$('<div></div>')
					.addClass('saving-message')
					.html(translate('webs.util.status.saving'))
					.append(top.$("<div></div>")
						.addClass('saving-spinner')
						.loadingSpinner()
					)
					.append(top.$("<div></div>")
						.addClass('saving-error')
						.html(translate('webs.util.error.UNABLE_TO_SAVE_REFRESH', {url: top.document.location.href}))
						.hide()
					)
				)
				.appendTo(top.$(top.window.document.body));
			window.onbeforeunload = function persist_onbeforeunload_stub(e){ return undefined; };
			pub.save = save = function disable_save_method() {
				log.trigger("persist", "error", "Save called while flushing save queue in onbeforeunload.");
			};
			var statusCheckInterval = setInterval(function checkSaveQueue(){
				if (saveQueue.length === 0) {
					clearInterval(statusCheckInterval);
					statusCheckInterval = undefined;
					if (intendedLinkTarget) {
						top.document.location.href = intendedLinkTarget;
					} else {
						top.document.location = "/s/manager/sitemanager";
					}
				}
			}, 200);
			setTimeout(function errorSaving(){
				$saveCover.find(".saving-error").show();
			}, 20000);
		},1);

		// WebKit
		return unsavedChangesWarning;
	};

	connectPrimus();

	return pub;
});
