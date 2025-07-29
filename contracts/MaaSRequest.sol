// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./MaaSRegistry.sol";

/**
 * @title MaaSRequest
 * @dev Manages travel requests in the MaaS ecosystem, including
 * creation, tracking, and status updates. Stores detailed request
 * information and enables rich queries.
 */
contract MaaSRequest {
    // ================ Enums ================
    
    /**
     * @dev Purpose of travel
     */
    enum Purpose { 
        Work,
        School,
        Shopping, 
        Medical, 
        Trip,
        Emergency,
        Leisure,
        Other
    }
    
    /**
     * @dev Status of the request
     */
    enum Status { 
        Active,          // Initial state, accepting offers
        Closed,          // No longer accepting offers
        Cancelled,       // Cancelled by commuter
        ServiceSelected, // Service provider selected
        InProgress,      // Service in progress
        Finished,        // Service completed
        Expired,         // Request expired
        Disputed         // Issue with service
    }
    
    /**
     * @dev Priority level for the request
     */
    enum Priority {
        Low,
        Standard,
        High,
        Urgent
    }
    
    // ================ Data Structures ================
    
    /**
     * @dev Main request data structure
     */
    struct Request {
        uint256 requestId;            // Unique ID
        uint256 commuterId;           // Commuter ID
        address owner;                // Request owner
        uint256[] origin;             // Origin coordinates [x,y]
        uint256[] destination;        // Destination coordinates [x,y]
        uint256 startTime;            // Requested start time
        Purpose travelPurpose;        // Purpose of travel
        Status status;                // Current status
        uint8 mode;                   // Transport mode (for specific requests)
        string flexibleTime;          // Flexibility description
        uint256 maxPrice;             // Maximum price commuter will pay
        uint256 createdAt;            // Creation timestamp
        uint256 lastUpdated;          // Last update timestamp
        Priority priority;            // Request priority
        uint256 travelerCount;        // Number of travelers
        bool recurring;               // Whether this is a recurring request
        uint256 recurringInterval;    // Interval for recurring requests
        uint256 recurringEndDate;     // End date for recurring requests
        string additionalInstructions; // Any special instructions
    }
    
    /**
     * @dev Requirements for a request
     */
    struct Requirements {
        string key;                   // Requirement type
        bool value;                   // Whether it's required
    }
    
    // ================ State Variables ================
    
    // Request storage
    mapping(uint256 => Request) public requests;
    
    // Requirements storage (requestId => array of requirements)
    mapping(uint256 => Requirements[]) public requestRequirements;
    
    // Tracking which requests belong to a commuter
    mapping(uint256 => uint256[]) public commuterRequests;
    
    // Request history for analytics
    struct RequestHistory {
        uint256 timestamp;
        Status oldStatus;
        Status newStatus;
        address updatedBy;
    }
    mapping(uint256 => RequestHistory[]) public requestHistory;
    
    // Counters and global variables
    uint256 public numberOfRequests = 0;
    uint256 public activeRequestCount = 0;
    mapping(uint8 => uint256) public requestsByPurpose;
    
    // Contract references
    MaaSRegistry public registry;
    address public admin;
    
    // ================ Events ================
    
    event RequestCreated(
        uint256 indexed requestId,
        uint256 indexed commuterId,
        uint256[] origin,
        uint256[] destination,
        uint256 startTime,
        Purpose travelPurpose,
        Status status,
        string flexibleTime
    );
    
    event RequestUpdated(
        uint256 indexed requestId,
        Status status,
        address updatedBy
    );
    
    event RequestCancelled(
        uint256 indexed requestId,
        uint256 timestamp,
        string reason
    );
    
    event RequestDisputed(
        uint256 indexed requestId,
        string reason,
        address disputedBy
    );
    
    event RequestResolved(
        uint256 indexed requestId,
        Status resolution,
        address resolvedBy
    );
    
    event RecurringRequestCreated(
        uint256 indexed baseRequestId,
        uint256 interval,
        uint256 endDate,
        uint256 numberOfRecurrences
    );
    
    // ================ Constructor ================
    
    constructor(address _registryAddress) {
        registry = MaaSRegistry(_registryAddress);
        admin = msg.sender;
    }
    
    // ================ Core Request Functions ================
    
    /**
     * @dev Create a new travel request
     * @param requestId Unique request ID
     * @param origin Origin coordinates [x,y]
     * @param destination Destination coordinates [x,y]
     * @param startTime Requested start time
     * @param travelPurpose Purpose of travel
     * @param flexibleTime Time flexibility description
     * @param requirementValues Boolean flags for requirements
     * @param requirementKeys Requirement type names
     * @param owner Address of request creator
     * @return Success status
     */
    function createRequest(
        uint256 requestId,
        uint256[] memory origin,
        uint256[] memory destination,
        uint256 startTime,
        Purpose travelPurpose,
        string memory flexibleTime,
        bool[] memory requirementValues,
        string[] memory requirementKeys,
        address owner
    ) public returns (bool) {
        // Verify the sender is a registered commuter
        (uint256 commuterId , , , , ,) = registry.getCommuter(owner);
        require(commuterId > 0, "Sender is not a registered commuter");
        
        // Create new request
        Request storage newRequest = requests[requestId];
        newRequest.requestId = requestId;
        newRequest.commuterId = commuterId;
        newRequest.owner = owner;
        newRequest.origin = origin;
        newRequest.destination = destination;
        newRequest.startTime = startTime;
        newRequest.travelPurpose = travelPurpose;
        newRequest.status = Status.Active;
        newRequest.flexibleTime = flexibleTime;
        newRequest.maxPrice = 0; // Default - can be updated later
        newRequest.createdAt = block.timestamp;
        newRequest.lastUpdated = block.timestamp;
        newRequest.priority = Priority.Standard; // Default priority
        newRequest.travelerCount = 1; // Default to single traveler
        newRequest.recurring = false;
        newRequest.recurringInterval = 0;
        newRequest.recurringEndDate = 0;
        newRequest.additionalInstructions = "";
        
        // Store requirements
        for (uint i = 0; i < requirementKeys.length; i++) {
            requestRequirements[requestId].push(Requirements({
                key: requirementKeys[i],
                value: requirementValues[i]
            }));
        }
        
        // Add to commuter's request list
        commuterRequests[commuterId].push(requestId);
        
        // Update counters
        numberOfRequests++;
        activeRequestCount++;
        requestsByPurpose[uint8(travelPurpose)]++;
        
        // Record initial history
        RequestHistory memory initialHistory = RequestHistory({
            timestamp: block.timestamp,
            oldStatus: Status.Active, // Initial status is same as new
            newStatus: Status.Active,
            updatedBy: owner
        });
        requestHistory[requestId].push(initialHistory);
        
        emit RequestCreated(
            requestId,
            commuterId,
            origin,
            destination,
            startTime,
            travelPurpose,
            Status.Active,
            flexibleTime
        );
        
        return true;
    }
    
    /**
     * @dev Create a recurring request series
     * @param baseRequestId ID for the first request
     * @param origin Origin coordinates [x,y]
     * @param destination Destination coordinates [x,y]
     * @param startTime First occurrence start time
     * @param travelPurpose Purpose of travel
     * @param flexibleTime Time flexibility
     * @param requirementValues Requirement flags
     * @param requirementKeys Requirement types
     * @param recurringInterval Seconds between recurrences
     * @param recurringEndDate End date for recurrence
     * @param maxRecurrences Maximum number of recurrences
     * @return Array of created request IDs
     */
    function createRecurringRequest(
        uint256 baseRequestId,
        uint256[] memory origin,
        uint256[] memory destination,
        uint256 startTime,
        Purpose travelPurpose,
        string memory flexibleTime,
        bool[] memory requirementValues,
        string[] memory requirementKeys,
        uint256 recurringInterval,
        uint256 recurringEndDate,
        uint8 maxRecurrences
    ) public returns (uint256[] memory) {
        require(recurringInterval > 0, "Interval must be positive");
        require(recurringEndDate > block.timestamp, "End date must be in future");
        require(maxRecurrences > 0 && maxRecurrences <= 50, "Invalid recurrence count");
        
        // Create first request
        createRequest(
            baseRequestId,
            origin,
            destination,
            startTime,
            travelPurpose,
            flexibleTime,
            requirementValues,
            requirementKeys,
            msg.sender
        );
        
        // Mark as recurring
        Request storage baseRequest = requests[baseRequestId];
        baseRequest.recurring = true;
        baseRequest.recurringInterval = recurringInterval;
        baseRequest.recurringEndDate = recurringEndDate;
        
        // Calculate number of recurrences
        uint256 timeSpan = recurringEndDate - startTime;
        uint8 recurrenceCount = uint8(timeSpan / recurringInterval);
        if (recurrenceCount > maxRecurrences) {
            recurrenceCount = maxRecurrences;
        }
        
        // Create subsequent requests
        uint256[] memory createdIds = new uint256[](recurrenceCount + 1);
        createdIds[0] = baseRequestId;
        
        for (uint8 i = 1; i <= recurrenceCount; i++) {
            uint256 newRequestId = uint256(keccak256(abi.encodePacked(baseRequestId, i)));
            uint256 newStartTime = startTime + (recurringInterval * i);
            
            // Skip if beyond end date
            if (newStartTime > recurringEndDate) {
                break;
            }
            
            createRequest(
                newRequestId,
                origin,
                destination,
                newStartTime,
                travelPurpose,
                flexibleTime,
                requirementValues,
                requirementKeys,
                msg.sender
            );
            
            // Link to base request
            Request storage newRequest = requests[newRequestId];
            newRequest.recurring = true;
            
            createdIds[i] = newRequestId;
        }
        
        emit RecurringRequestCreated(baseRequestId, recurringInterval, recurringEndDate, recurrenceCount);
        
        return createdIds;
    }
    
    /**
     * @dev Update a request's status
     * @param requestId Request ID
     * @param newStatus New status
     * @return Success status
     */
    function updateRequestStatus(uint256 requestId, Status newStatus) public returns (bool) {
        Request storage request = requests[requestId];
        
        // Verify the request exists
        require(request.requestId == requestId, "Request does not exist");
        
        // Verify sender's authority
        bool isAuthorized = false;
        
        // Commuter who created the request
        if (request.owner == msg.sender) {
            isAuthorized = true;
        }
        
        // Service provider assigned to request
        else if (newStatus == Status.InProgress || newStatus == Status.Finished) {
            // This would require checking if sender is the assigned provider
            // For simplicity, admin can update to these statuses
            isAuthorized = (msg.sender == admin);
        }
        
        // Admin can always update
        else if (msg.sender == admin) {
            isAuthorized = true;
        }
        
        require(isAuthorized, "Not authorized to update status");
        
        // Get current status for history
        Status oldStatus = request.status;
        
        // Update status
        request.status = newStatus;
        request.lastUpdated = block.timestamp;
        
        // Update active request count
        if (oldStatus == Status.Active && newStatus != Status.Active) {
            activeRequestCount--;
        } else if (oldStatus != Status.Active && newStatus == Status.Active) {
            activeRequestCount++;
        }
        
        // Record history
        RequestHistory memory historyEntry = RequestHistory({
            timestamp: block.timestamp,
            oldStatus: oldStatus,
            newStatus: newStatus,
            updatedBy: msg.sender
        });
        requestHistory[requestId].push(historyEntry);
        
        emit RequestUpdated(requestId, newStatus, msg.sender);
        
        return true;
    }
    
    /**
     * @dev Cancel a request
     * @param requestId Request ID
     * @param reason Cancellation reason
     * @return Success status
     */
    function cancelRequest(uint256 requestId, string memory reason) public returns (bool) {
        Request storage request = requests[requestId];
        
        // Verify the request exists
        require(request.requestId == requestId, "Request does not exist");
        
        // Verify sender owns the request or is admin
        require(request.owner == msg.sender || msg.sender == admin, 
                "Not authorized to cancel");
        
        // Verify request is in a cancellable state
        require(request.status == Status.Active || request.status == Status.ServiceSelected, 
                "Request cannot be cancelled in current state");
        
        // Get current status for history
        Status oldStatus = request.status;
        
        // Update status
        request.status = Status.Cancelled;
        request.lastUpdated = block.timestamp;
        
        // Update active request count if needed
        if (oldStatus == Status.Active) {
            activeRequestCount--;
        }
        
        // Record history
        RequestHistory memory historyEntry = RequestHistory({
            timestamp: block.timestamp,
            oldStatus: oldStatus,
            newStatus: Status.Cancelled,
            updatedBy: msg.sender
        });
        requestHistory[requestId].push(historyEntry);
        
        emit RequestCancelled(requestId, block.timestamp, reason);
        
        return true;
    }
    
    /**
     * @dev Open a dispute for a request
     * @param requestId Request ID
     * @param reason Dispute reason
     * @return Success status
     */
    function disputeRequest(uint256 requestId, string memory reason) public returns (bool) {
        Request storage request = requests[requestId];
        
        // Verify the request exists
        require(request.requestId == requestId, "Request does not exist");
        
        // Verify sender is authorized (owner or provider)
        bool isOwner = (request.owner == msg.sender);
        
        // TODO: Verify if sender is the assigned provider
        bool isProvider = false; // Placeholder - needs implementation
        
        require(isOwner || isProvider || msg.sender == admin, "Not authorized to dispute");
        
        // Verify request is in a disputable state
        require(request.status == Status.ServiceSelected || 
                request.status == Status.InProgress || 
                request.status == Status.Finished,
                "Request cannot be disputed in current state");
        
        // Get current status for history
        Status oldStatus = request.status;
        
        // Update status
        request.status = Status.Disputed;
        request.lastUpdated = block.timestamp;
        
        // Record history
        RequestHistory memory historyEntry = RequestHistory({
            timestamp: block.timestamp,
            oldStatus: oldStatus,
            newStatus: Status.Disputed,
            updatedBy: msg.sender
        });
        requestHistory[requestId].push(historyEntry);
        
        emit RequestDisputed(requestId, reason, msg.sender);
        
        return true;
    }
    
    /**
     * @dev Resolve a disputed request
     * @param requestId Request ID
     * @param resolution Resolution status
     * @return Success status
     */
    function resolveDispute(uint256 requestId, Status resolution) public returns (bool) {
        Request storage request = requests[requestId];
        
        // Verify the request exists and is disputed
        require(request.requestId == requestId, "Request does not exist");
        require(request.status == Status.Disputed, "Request is not disputed");
        
        // Only admin can resolve disputes
        require(msg.sender == admin, "Only admin can resolve disputes");
        
        // Verify resolution is valid
        require(resolution == Status.Finished || 
                resolution == Status.Cancelled,
                "Invalid resolution status");
        
        // Update status
        request.status = resolution;
        request.lastUpdated = block.timestamp;
        
        // Record history
        RequestHistory memory historyEntry = RequestHistory({
            timestamp: block.timestamp,
            oldStatus: Status.Disputed,
            newStatus: resolution,
            updatedBy: msg.sender
        });
        requestHistory[requestId].push(historyEntry);
        
        emit RequestResolved(requestId, resolution, msg.sender);
        
        return true;
    }
    
    // ================ Request Detail Functions ================
    
    /**
     * @dev Update request details
     * @param requestId Request ID
     * @param maxPrice Maximum price
     * @param priority Priority level
     * @param travelerCount Number of travelers
     * @param additionalInstructions Special instructions
     * @return Success status
     */
    function updateRequestDetails(
        uint256 requestId,
        uint256 maxPrice,
        Priority priority,
        uint256 travelerCount,
        string memory additionalInstructions
    ) public returns (bool) {
        Request storage request = requests[requestId];
        
        // Verify the request exists
        require(request.requestId == requestId, "Request does not exist");
        
        // Verify sender owns the request
        require(request.owner == msg.sender, "Not authorized to update");
        
        // Verify request is in updatable state
        require(request.status == Status.Active, "Request cannot be updated in current state");
        
        // Update details
        request.maxPrice = maxPrice;
        request.priority = priority;
        request.travelerCount = travelerCount;
        request.additionalInstructions = additionalInstructions;
        request.lastUpdated = block.timestamp;
        
        return true;
    }
    
    /**
     * @dev Add a requirement to a request
     * @param requestId Request ID
     * @param key Requirement type
     * @param value Requirement value
     * @return Success status
     */
    function addRequirement(uint256 requestId, string memory key, bool value) public returns (bool) {
        Request storage request = requests[requestId];
        
        // Verify the request exists
        require(request.requestId == requestId, "Request does not exist");
        
        // Verify sender owns the request
        require(request.owner == msg.sender, "Not authorized to update");
        
        // Verify request is in updatable state
        require(request.status == Status.Active, "Request cannot be updated in current state");
        
        // Add requirement
        requestRequirements[requestId].push(Requirements({
            key: key,
            value: value
        }));
        
        return true;
    }
    
    // ================ Query Functions ================
    
    /**
     * @dev Get basic info about a request
     * @param requestId Request ID
     * @return requestId, commuterId, origin, destination, startTime, travelPurpose, status, mode
     */
    function getRequestBasicInfo(uint256 requestId) public view returns (
        uint256, uint256, uint256[] memory, uint256[] memory, uint256, Purpose, Status, uint8
    ) {
        Request storage request = requests[requestId];
        return (
            request.requestId,
            request.commuterId,
            request.origin,
            request.destination,
            request.startTime,
            request.travelPurpose,
            request.status,
            request.mode
        );
    }
    
    /**
     * @dev Get detailed info about a request
     * @param requestId Request ID
     * @return Full request struct
     */
    function getRequestDetails(uint256 requestId) public view returns (Request memory) {
        return requests[requestId];
    }
    
    /**
     * @dev Check if a request has a specific requirement
     * @param requestId Request ID
     * @param key Requirement key
     * @return Requirement value or false if not found
     */
    function hasRequirement(uint256 requestId, string memory key) public view returns (bool) {
        Requirements[] storage reqs = requestRequirements[requestId];
        for (uint256 i = 0; i < reqs.length; i++) {
            if (keccak256(bytes(reqs[i].key)) == keccak256(bytes(key))) {
                return reqs[i].value;
            }
        }
        return false;
    }
    
    /**
     * @dev Get all requirements for a request
     * @param requestId Request ID
     * @return Array of requirements
     */
    function getRequirements(uint256 requestId) public view returns (Requirements[] memory) {
        return requestRequirements[requestId];
    }
    
    /**
     * @dev Get all requests for a commuter
     * @param commuterId Commuter ID
     * @return Array of request IDs
     */
    function getCommuterRequests(uint256 commuterId) public view returns (uint256[] memory) {
        return commuterRequests[commuterId];
    }
    
    /**
     * @dev Get request history
     * @param requestId Request ID
     * @return Array of history entries
     */
    function getRequestHistory(uint256 requestId) public view returns (RequestHistory[] memory) {
        return requestHistory[requestId];
    }
    
    /**
     * @dev Get active request count by purpose
     * @param purpose Travel purpose
     * @return Count of active requests for that purpose
     */
    function getActiveRequestCountByPurpose(Purpose purpose) public view returns (uint256) {
        return requestsByPurpose[uint8(purpose)];
    }
    
    /**
     * @dev Get active request IDs
     * @param startIndex Pagination start
     * @param count Number of records to return
     * @return Array of active request IDs
     */
    function getActiveRequestIds(uint256 startIndex, uint256 count) public view returns (uint256[] memory) {
        // This is computationally expensive, but included for completeness
        // In a production system, you'd maintain a separate array of active requests
        
        uint256[] memory activeIds = new uint256[](count);
        uint256 foundCount = 0;
        uint256 totalChecked = 0;
        
        for (uint256 i = 1; i <= numberOfRequests && foundCount < count; i++) {
            if (requests[i].status == Status.Active) {
                if (totalChecked >= startIndex) {
                    activeIds[foundCount] = i;
                    foundCount++;
                }
                totalChecked++;
            }
        }
        
        return activeIds;
    }
    
    /**
     * @dev Get active request count
     * @return Count of active requests
     */
    function getActiveRequestCount() public view returns (uint256) {
        return activeRequestCount;
    }
}