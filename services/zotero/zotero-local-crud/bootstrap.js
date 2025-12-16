/**
 * Zotero Local CRUD API
 *
 * A Zotero 7 bootstrapped plugin that exposes HTTP endpoints for full CRUD
 * operations on the local Zotero library without requiring cloud sync.
 *
 * Endpoints:
 *   GET  /local-crud/ping           - Health check
 *   POST /local-crud/items          - Create item
 *   GET  /local-crud/item?key=...   - Read item
 *   PATCH /local-crud/item?key=...  - Update item
 *   DELETE /local-crud/item?key=... - Delete item
 *   POST /local-crud/search         - Search items
 */

var LocalCrudAPI;

// Plugin lifecycle hooks
function startup({ id, version, rootURI }, reason) {
    LocalCrudAPI = {
        id,
        version,
        rootURI
    };

    registerEndpoints();
    Zotero.debug("Local CRUD API: Started (v" + version + ")");
}

function shutdown({ id, version, rootURI }, reason) {
    unregisterEndpoints();
    LocalCrudAPI = null;
    Zotero.debug("Local CRUD API: Shutdown");
}

function install(data, reason) {
    Zotero.debug("Local CRUD API: Installed");
}

function uninstall(data, reason) {
    Zotero.debug("Local CRUD API: Uninstalled");
}

// Endpoint registration
function registerEndpoints() {
    Zotero.Server.Endpoints["/local-crud/ping"] = PingEndpoint;
    Zotero.Server.Endpoints["/local-crud/items"] = CreateItemEndpoint;
    Zotero.Server.Endpoints["/local-crud/item"] = ItemEndpoint;
    Zotero.Server.Endpoints["/local-crud/search"] = SearchEndpoint;
    Zotero.debug("Local CRUD API: Registered 4 endpoints");
}

function unregisterEndpoints() {
    delete Zotero.Server.Endpoints["/local-crud/ping"];
    delete Zotero.Server.Endpoints["/local-crud/items"];
    delete Zotero.Server.Endpoints["/local-crud/item"];
    delete Zotero.Server.Endpoints["/local-crud/search"];
    Zotero.debug("Local CRUD API: Unregistered endpoints");
}

// Helper: Send JSON response
function jsonResponse(status, data) {
    return [status, "application/json", JSON.stringify(data)];
}

// Helper: Parse JSON body safely
function parseJSON(data) {
    if (!data) return {};
    if (typeof data === 'object') return data;
    try {
        return JSON.parse(data);
    } catch (e) {
        return null;
    }
}

// Helper: Serialize item to JSON
function serializeItem(item) {
    var data = {
        key: item.key,
        itemID: item.id,
        itemType: Zotero.ItemTypes.getName(item.itemTypeID),
        version: item.version,
        libraryID: item.libraryID,
        dateAdded: item.dateAdded,
        dateModified: item.dateModified,
        fields: {},
        creators: item.getCreatorsJSON(),
        tags: item.getTags(),
        collections: item.getCollections()
    };

    // Get all fields for this item type
    var fieldIDs = Zotero.ItemFields.getItemTypeFields(item.itemTypeID);
    for (let fieldID of fieldIDs) {
        var fieldName = Zotero.ItemFields.getName(fieldID);
        var value = item.getField(fieldName);
        if (value) {
            data.fields[fieldName] = value;
        }
    }

    // Also get base fields that might not be in type-specific list
    var baseFields = ['title', 'abstractNote', 'url', 'accessDate', 'date', 'extra'];
    for (let fieldName of baseFields) {
        try {
            var value = item.getField(fieldName);
            if (value && !data.fields[fieldName]) {
                data.fields[fieldName] = value;
            }
        } catch (e) {
            // Field not valid for this item type
        }
    }

    return data;
}

/**
 * GET /local-crud/ping
 * Health check endpoint
 */
var PingEndpoint = function() {};
PingEndpoint.prototype = {
    supportedMethods: ["GET"],
    supportedDataTypes: ["application/json"],
    permitBookmarklet: false,

    init: async function(request) {
        return jsonResponse(200, {
            status: "ok",
            plugin: "zotero-local-crud",
            version: LocalCrudAPI.version,
            zoteroVersion: Zotero.version,
            timestamp: new Date().toISOString(),
            libraryID: Zotero.Libraries.userLibraryID
        });
    }
};

/**
 * POST /local-crud/items
 * Create a new item
 *
 * Request body:
 * {
 *   "itemType": "book",
 *   "fields": { "title": "Example", "date": "2024" },
 *   "creators": [{ "firstName": "John", "lastName": "Doe", "creatorType": "author" }],
 *   "tags": ["tag1", "tag2"] or [{ "tag": "tag1", "type": 0 }],
 *   "collections": ["ABCD1234"]
 * }
 */
var CreateItemEndpoint = function() {};
CreateItemEndpoint.prototype = {
    supportedMethods: ["POST"],
    supportedDataTypes: ["application/json"],
    permitBookmarklet: false,

    init: async function(request) {
        try {
            var data = parseJSON(request.data);

            if (data === null) {
                return jsonResponse(400, { error: "Invalid JSON in request body" });
            }

            if (!data.itemType) {
                return jsonResponse(400, { error: "itemType is required" });
            }

            // Validate item type
            var itemTypeID = Zotero.ItemTypes.getID(data.itemType);
            if (!itemTypeID) {
                return jsonResponse(400, {
                    error: "Invalid itemType: " + data.itemType,
                    validTypes: Zotero.ItemTypes.getTypes().map(t => t.name)
                });
            }

            // Create new item
            var item = new Zotero.Item(data.itemType);
            item.libraryID = Zotero.Libraries.userLibraryID;

            // Set fields
            if (data.fields) {
                for (let [field, value] of Object.entries(data.fields)) {
                    try {
                        item.setField(field, value);
                    } catch (e) {
                        Zotero.debug("Local CRUD API: Could not set field " + field + ": " + e.message);
                    }
                }
            }

            // Set creators
            if (data.creators && Array.isArray(data.creators)) {
                item.setCreators(data.creators);
            }

            // Set tags
            if (data.tags && Array.isArray(data.tags)) {
                for (let tag of data.tags) {
                    if (typeof tag === 'string') {
                        item.addTag(tag);
                    } else if (tag.tag) {
                        item.addTag(tag.tag, tag.type || 0);
                    }
                }
            }

            // Set collections
            if (data.collections && Array.isArray(data.collections)) {
                // Convert collection keys to IDs
                var collectionIDs = [];
                for (let key of data.collections) {
                    var collection = Zotero.Collections.getByLibraryAndKey(
                        Zotero.Libraries.userLibraryID,
                        key
                    );
                    if (collection) {
                        collectionIDs.push(collection.id);
                    }
                }
                if (collectionIDs.length > 0) {
                    item.setCollections(collectionIDs);
                }
            }

            // Save item
            await item.saveTx();

            Zotero.debug("Local CRUD API: Created item " + item.key);

            return jsonResponse(201, {
                key: item.key,
                itemID: item.id,
                version: item.version,
                itemType: data.itemType
            });

        } catch (e) {
            Zotero.debug("Local CRUD API Error (create): " + e.message);
            return jsonResponse(500, { error: e.message });
        }
    }
};

/**
 * GET/PATCH/DELETE /local-crud/item?key=XXXXXXXX
 * Read, update, or delete an item
 */
var ItemEndpoint = function() {};
ItemEndpoint.prototype = {
    supportedMethods: ["GET", "PATCH", "DELETE"],
    supportedDataTypes: ["application/json"],
    permitBookmarklet: false,

    init: async function(request) {
        try {
            // Get key from query params
            var key = request.query?.key;

            if (!key) {
                return jsonResponse(400, { error: "key parameter is required" });
            }

            // Get item by key
            var libraryID = Zotero.Libraries.userLibraryID;
            var item = await Zotero.Items.getByLibraryAndKeyAsync(libraryID, key);

            if (!item) {
                return jsonResponse(404, { error: "Item not found", key: key });
            }

            // Route to appropriate handler
            switch (request.method) {
                case "GET":
                    return this.handleGet(item);
                case "PATCH":
                    return await this.handlePatch(item, request.data);
                case "DELETE":
                    return await this.handleDelete(item);
                default:
                    return jsonResponse(405, { error: "Method not allowed" });
            }

        } catch (e) {
            Zotero.debug("Local CRUD API Error (item): " + e.message);
            return jsonResponse(500, { error: e.message });
        }
    },

    handleGet: function(item) {
        return jsonResponse(200, serializeItem(item));
    },

    handlePatch: async function(item, requestData) {
        var updates = parseJSON(requestData);

        if (updates === null) {
            return jsonResponse(400, { error: "Invalid JSON in request body" });
        }

        // Update fields
        if (updates.fields) {
            for (let [field, value] of Object.entries(updates.fields)) {
                try {
                    item.setField(field, value);
                } catch (e) {
                    Zotero.debug("Local CRUD API: Could not update field " + field + ": " + e.message);
                }
            }
        }

        // Update creators (replace all)
        if (updates.creators && Array.isArray(updates.creators)) {
            item.setCreators(updates.creators);
        }

        // Update tags (replace all)
        if (updates.tags !== undefined) {
            // Remove existing tags
            var existingTags = item.getTags();
            for (let tag of existingTags) {
                item.removeTag(tag.tag);
            }
            // Add new tags
            if (Array.isArray(updates.tags)) {
                for (let tag of updates.tags) {
                    if (typeof tag === 'string') {
                        item.addTag(tag);
                    } else if (tag.tag) {
                        item.addTag(tag.tag, tag.type || 0);
                    }
                }
            }
        }

        // Update collections (replace all)
        if (updates.collections !== undefined) {
            var collectionIDs = [];
            if (Array.isArray(updates.collections)) {
                for (let key of updates.collections) {
                    var collection = Zotero.Collections.getByLibraryAndKey(
                        Zotero.Libraries.userLibraryID,
                        key
                    );
                    if (collection) {
                        collectionIDs.push(collection.id);
                    }
                }
            }
            item.setCollections(collectionIDs);
        }

        await item.saveTx();

        Zotero.debug("Local CRUD API: Updated item " + item.key);

        return jsonResponse(200, {
            key: item.key,
            version: item.version,
            dateModified: item.dateModified
        });
    },

    handleDelete: async function(item) {
        var key = item.key;
        await item.eraseTx();

        Zotero.debug("Local CRUD API: Deleted item " + key);

        return [204, "application/json", ""];
    }
};

/**
 * POST /local-crud/search
 * Search for items
 *
 * Request body:
 * {
 *   "conditions": [
 *     { "condition": "title", "operator": "contains", "value": "example" },
 *     { "condition": "tag", "operator": "is", "value": "mytag" }
 *   ],
 *   "limit": 100,
 *   "includeFullData": false
 * }
 *
 * Condition operators: is, isNot, contains, doesNotContain, isLessThan, isGreaterThan, isBefore, isAfter, etc.
 * Special conditions: quicksearch-everything, quicksearch-titleCreatorYear
 */
var SearchEndpoint = function() {};
SearchEndpoint.prototype = {
    supportedMethods: ["POST"],
    supportedDataTypes: ["application/json"],
    permitBookmarklet: false,

    init: async function(request) {
        try {
            var data = parseJSON(request.data);

            if (data === null) {
                return jsonResponse(400, { error: "Invalid JSON in request body" });
            }

            var conditions = data.conditions || [];
            var limit = data.limit || 100;
            var includeFullData = data.includeFullData || false;

            // Create search object
            var search = new Zotero.Search();
            search.libraryID = Zotero.Libraries.userLibraryID;

            // Handle empty conditions (return all items)
            if (conditions.length === 0) {
                // Just search with no conditions to get all items
                search.addCondition('itemType', 'isNot', 'attachment');
                search.addCondition('itemType', 'isNot', 'note');
            } else {
                // Add search conditions
                for (let cond of conditions) {
                    if (!cond.condition) continue;

                    var operator = cond.operator || 'contains';
                    var value = cond.value || '';
                    var required = cond.required !== false;

                    search.addCondition(cond.condition, operator, value, required);
                }
            }

            // Execute search
            var itemIDs = await search.search();

            // Apply limit
            if (itemIDs.length > limit) {
                itemIDs = itemIDs.slice(0, limit);
            }

            // Get items
            var items = await Zotero.Items.getAsync(itemIDs);

            // Serialize results
            var results;
            if (includeFullData) {
                results = items.map(serializeItem);
            } else {
                results = items.map(item => ({
                    key: item.key,
                    itemID: item.id,
                    itemType: Zotero.ItemTypes.getName(item.itemTypeID),
                    title: item.getField('title') || '',
                    dateModified: item.dateModified
                }));
            }

            Zotero.debug("Local CRUD API: Search returned " + results.length + " items");

            return jsonResponse(200, {
                total: results.length,
                limit: limit,
                items: results
            });

        } catch (e) {
            Zotero.debug("Local CRUD API Error (search): " + e.message);
            return jsonResponse(500, { error: e.message });
        }
    }
};
