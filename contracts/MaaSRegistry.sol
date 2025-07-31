// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title MaaSRegistry
 * @dev Central registry for commuters and service providers in the MaaS ecosystem
 * Enhanced with reputation systems, advanced profiles, and integration with rewards
 */
contract MaaSRegistry {
    // ================ Enumerations ================
    
    enum TransportMode { BackgroundTraffic, Train, Bus, Car, Bike, Walk, MaaS }
    enum VerificationLevel { None, Basic, Advanced, Validated }

    // ================ Data Structures ================
    
    /**
     * @dev Minimal commuter profile - personal data moved off-chain
     */
    struct Commuter {
        uint256 commuterId;
        uint256 registrationTime;        // When the commuter registered
        uint256 lastActiveTime;          // Last activity timestamp
        uint256 tripCount;               // Number of completed trips
        uint256 totalSpent;              // Total amount spent on services
        uint256 reputationScore;         // Overall reputation score (0-100)
        bool isActive;                   // Whether commuter is active
    }

    /**
     * @dev Minimal provider profile - only essential consensus data
     */
    struct Provider {
        uint256 providerId;
        address providerAddress;
        string companyName;
        TransportMode modeType;
        uint256 basePrice;        // In wei
        uint256 capacity;
        bool isActive;           // Can be toggled on/off
    }

    
    /**
     * @dev Reputation feedback submitted by users
     */
    struct Feedback {
        uint256 feedbackId;
        uint256 tripId;                   // Associated trip/service
        address submitter;                // Who submitted the feedback
        address subject;                  // About whom (commuter or provider)
        uint256 rating;                   // Rating score (0-100)
        string comment;                   // Feedback text
        uint256 timestamp;                // When feedback was submitted
        bool isVerified;                  // Whether feedback is verified
    }
    
    /**
     * @dev Public profile information visible to all
     */
    struct PublicProfile {
        uint256 id;                       // Commuter or provider ID
        string displayName;               // Public display name
        string bio;                       // Short biography/description
        string avatarURI;                 // URI for avatar image
        uint256 joinDate;                 // When they joined the platform
        uint256 reputationScore;          // Public reputation score
        bool isVerified;                  // Verification status
        string[] badges;                  // Achievement badges
    }

    // ================ State Variables ================
    
    // Mapping from address to commuter/provider data
    mapping(address => Commuter) private commuters;
    mapping(address => Provider) private providers;
    
    // Mapping from IDs to addresses
    mapping(uint256 => address) public commuterIdToAddress;
    mapping(uint256 => address) public providerIdToAddress;
    
    // Counters
    uint256 public numberOfCommuters = 0;
    uint256 public numberOfProviders = 0;
    
    // Reputation system
    mapping(address => mapping(address => Feedback[])) private feedbackHistory; // subject => submitter => feedbacks
    mapping(address => uint256) public reputationScores;
    mapping(address => uint256) public feedbackCount;
    
    // Public profiles
    mapping(address => PublicProfile) public publicProfiles;
    
    // Reward system integration
    mapping(address => uint256) public loyaltyPoints;
    
    // Platform governance
    address public admin;
    mapping(address => bool) public verifiers;
    bool public registrationPaused = false;
    
    // ================ Events ================
    
    event CommuterAdded(
    uint256 indexed commuterId,
    address indexed commuter
    );

    event ProviderAdded(
        uint256 providerId,
        address providerAddress,
        string companyName,
        TransportMode modeType,
        uint256 basePrice,
        uint256 capacity
    );
    
    event ProfileUpdated(address indexed account, string profileType);
    event FeedbackSubmitted(address indexed subject, address indexed submitter, uint256 rating);
    event ReputationUpdated(address indexed account, uint256 newScore);
    event UserVerified(address indexed account, VerificationLevel level);
    event LoyaltyPointsAwarded(address indexed user, uint256 points, string reason);
    
    // ================ Constructor ================
    
    constructor() {
        admin = msg.sender;
        verifiers[admin] = true;
    }
    
    // ================ Modifier ================
    
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can call this function");
        _;
    }
    
    modifier onlyVerifier() {
        require(verifiers[msg.sender], "Only verifiers can call this function");
        _;
    }
    
    modifier registrationActive() {
        require(!registrationPaused, "Registration is currently paused");
        _;
    }
    
    // ================ Registration Functions ================
    
    /**
     * @dev Register a new commuter with minimal data
     */
    function addCommuter(uint256 commuterId) public registrationActive {
        // Verify address not already registered
        require(commuters[msg.sender].commuterId == 0, "Address already registered");
        
        // Create minimal commuter record
        commuters[msg.sender].commuterId = commuterId;
        commuters[msg.sender].registrationTime = block.timestamp;
        commuters[msg.sender].lastActiveTime = block.timestamp;
        commuters[msg.sender].tripCount = 0;
        commuters[msg.sender].totalSpent = 0;
        commuters[msg.sender].reputationScore = 50; // Default neutral score
        commuters[msg.sender].isActive = true;
        
        // Map ID to address
        commuterIdToAddress[commuterId] = msg.sender;
        numberOfCommuters++;
        
        // Award welcome points
        awardLoyaltyPoints(msg.sender, 10, "welcome_bonus");
        
        emit CommuterAdded(commuterId, msg.sender);
    }

    /**
     * @dev Register a new service provider with minimal on-chain data
     * @param providerId Unique provider ID
     * @param companyName Provider company name
     * @param modeType Transport mode enum
     * @param basePrice Base price in wei
     * @param capacity Service capacity
     */
    function addProvider(
        uint256 providerId, 
        string memory companyName,
        TransportMode modeType,
        uint256 basePrice,
        uint256 capacity
    ) public registrationActive {
        require(providers[msg.sender].providerId == 0, "Address already registered");
        
        // Store minimal provider data
        providers[msg.sender] = Provider({
            providerId: providerId,
            providerAddress: msg.sender,
            companyName: companyName,
            modeType: modeType,
            basePrice: basePrice,
            capacity: capacity,
            isActive: true
        });
        
        // Map ID to address
        providerIdToAddress[providerId] = msg.sender;
        numberOfProviders++;
        
        // Create minimal public profile
        publicProfiles[msg.sender] = PublicProfile({
            id: providerId,
            displayName: companyName,
            bio: "",
            avatarURI: "",
            joinDate: block.timestamp,
            reputationScore: 70,
            isVerified: false,
            badges: new string[](0)
        });
        
        emit ProviderAdded(
            providerId,
            msg.sender,
            companyName,
            modeType,
            basePrice,
            capacity
        );
    }
    
    // ================ Profile Management ================

    /**
     * @dev Update provider total capacity (less frequent, on-chain)
     * Available capacity should be managed off-chain for efficiency
     */
    function updateProviderCapacity(uint256 newCapacity) public {
        require(providers[msg.sender].providerId > 0, "Not a registered provider");
        providers[msg.sender].capacity = newCapacity;
    }
    
    /**
     * @dev Update provider base price (frequently changed data)
     */
    function updateProviderPrice(uint256 newBasePrice) public {
        require(providers[msg.sender].providerId > 0, "Not a registered provider");
        providers[msg.sender].basePrice = newBasePrice;
    }
        
    /**
     * @dev Update public profile
     * @param displayName New display name
     * @param bio New biography
     * @param avatarURI New avatar URI
     */
    function updatePublicProfile(string memory displayName, string memory bio, string memory avatarURI) public {
        require(commuters[msg.sender].commuterId > 0 || providers[msg.sender].providerId > 0, 
                "Not registered");
        
        publicProfiles[msg.sender].displayName = displayName;
        publicProfiles[msg.sender].bio = bio;
        publicProfiles[msg.sender].avatarURI = avatarURI;
        
        emit ProfileUpdated(msg.sender, commuters[msg.sender].commuterId > 0 ? "commuter" : "provider");
    }

    /**
     * @dev Toggle provider active status
     */
    function setProviderActive(bool isActive) public {
        require(providers[msg.sender].providerId > 0, "Not a registered provider");
        providers[msg.sender].isActive = isActive;
    }
    
    
    // ================ Reputation System ================
    
    /**
     * @dev Submit feedback about a commuter or provider
     * @param subject Address receiving feedback
     * @param tripId Related trip/service ID
     * @param rating Rating score (0-100)
     * @param comment Feedback comment
     */
    function submitFeedback(address subject, uint256 tripId, uint256 rating, string memory comment) public {
        require(rating <= 100, "Rating must be 0-100");
        require(commuters[subject].commuterId > 0 || providers[subject].providerId > 0, 
                "Subject is not registered");
        require(commuters[msg.sender].commuterId > 0 || providers[msg.sender].providerId > 0, 
                "Submitter is not registered");
        require(subject != msg.sender, "Cannot rate yourself");
        
        // Create feedback
        Feedback memory newFeedback = Feedback({
            feedbackId: feedbackCount[subject],
            tripId: tripId,
            submitter: msg.sender,
            subject: subject,
            rating: rating,
            comment: comment,
            timestamp: block.timestamp,
            isVerified: false
        });
        
        // Store feedback
        feedbackHistory[subject][msg.sender].push(newFeedback);
        feedbackCount[subject]++;
        
        // Update reputation score
        updateReputationScore(subject);
        
        // Award feedback points
        awardLoyaltyPoints(msg.sender, 5, "feedback_submitted");
        
        emit FeedbackSubmitted(subject, msg.sender, rating);
    }
    
    /**
     * @dev Verify feedback as legitimate
     * @param subject Subject of feedback
     * @param submitter Submitter of feedback
     * @param feedbackId ID of feedback to verify
     */
    function verifyFeedback(address subject, address submitter, uint256 feedbackId) public onlyVerifier {
        require(feedbackId < feedbackHistory[subject][submitter].length, "Feedback does not exist");
        
        feedbackHistory[subject][submitter][feedbackId].isVerified = true;
        
        // Update reputation with verified feedback weighted more heavily
        updateReputationScore(subject);
    }
    
    /**
     * @dev Calculate and update reputation score for an address
     * @param account Address to update score for
     */
    function updateReputationScore(address account) internal {
        uint256 totalRating = 0;
        uint256 feedbacksReceived = 0;
        
        // We need to iterate through all possible submitters
        // This isn't scalable but simplifies implementation for this example
        for (uint256 i = 1; i <= numberOfCommuters + numberOfProviders; i++) {
            address submitter;
            if (i <= numberOfCommuters) {
                submitter = commuterIdToAddress[i];
            } else {
                submitter = providerIdToAddress[i - numberOfCommuters];
            }
            
            if (submitter == address(0)) continue;
            
            Feedback[] storage feedbacks = feedbackHistory[account][submitter];
            for (uint256 j = 0; j < feedbacks.length; j++) {
                uint256 weight = feedbacks[j].isVerified ? 2 : 1; // Verified feedback counts double
                totalRating += feedbacks[j].rating * weight;
                feedbacksReceived += weight;
            }
        }
        
        // Calculate new score, default to 50 if no feedback
        uint256 newScore = feedbacksReceived > 0 ? totalRating / feedbacksReceived : 50;
        
        // Update scores
        reputationScores[account] = newScore;
        publicProfiles[account].reputationScore = newScore;
        
        if (commuters[account].commuterId > 0) {
            commuters[account].reputationScore = newScore;
        }
        
        emit ReputationUpdated(account, newScore);
    }
    
    // ================ Usage Tracking ================
    
    /**
     * @dev Record a completed trip for a commuter
     * Simplified version - detailed analytics should be handled off-chain
     * @param commuter Commuter address
     * @param amount Amount spent
     */
    function recordTrip(address commuter, uint256 amount) public {
        require(msg.sender == admin || verifiers[msg.sender], "Not authorized to record trips");
        require(commuters[commuter].commuterId > 0, "Not a registered commuter");
        
        commuters[commuter].tripCount++;
        commuters[commuter].totalSpent += amount;
        commuters[commuter].lastActiveTime = block.timestamp;
        
        // Award trip points based on amount
        uint256 points = amount / 10 + 1;
        awardLoyaltyPoints(commuter, points, "trip_completed");
    }
    
    /**
     * @dev Record basic service activity for a provider
     * Detailed service analytics (revenue, performance metrics) should be handled off-chain
     * This only tracks basic activity for reputation purposes
     * @param provider Provider address
     */
    function recordServiceActivity(address provider) public {
        require(msg.sender == admin || verifiers[msg.sender], "Not authorized to record services");
        require(providers[provider].providerId > 0, "Not a registered provider");
        
        // Only award loyalty points - detailed tracking should be off-chain
        awardLoyaltyPoints(provider, 2, "service_provided");
    }
    
    // ================ Reward System ================
    
    /**
     * @dev Award loyalty points to a user
     * @param user User address
     * @param points Points to award
     * @param reason Reason for points
     */
    function awardLoyaltyPoints(address user, uint256 points, string memory reason) internal {
        loyaltyPoints[user] += points;
        emit LoyaltyPointsAwarded(user, points, reason);
    }
    
    /**
     * @dev External function to award loyalty points (admin only)
     * @param user User address
     * @param points Points to award
     * @param reason Reason for points
     */
    function awardPoints(address user, uint256 points, string memory reason) public onlyAdmin {
        awardLoyaltyPoints(user, points, reason);
    }
    
    /**
     * @dev Get loyalty points balance
     * @param user User address
     * @return Points balance
     */
    function getPoints(address user) public view returns (uint256) {
        return loyaltyPoints[user];
    }
    
    // ================ Admin Functions ================
    
    /**
     * @dev Pause or unpause registration
     * @param paused Whether registration should be paused
     */
    function setRegistrationPaused(bool paused) public onlyAdmin {
        registrationPaused = paused;
    }
    
    /**
     * @dev Change admin
     * @param newAdmin New admin address
     */
    function changeAdmin(address newAdmin) public onlyAdmin {
        require(newAdmin != address(0), "Invalid address");
        admin = newAdmin;
        verifiers[newAdmin] = true;
    }
    
    /**
     * @dev Emergency remove user
     * @param user Address to remove
     */
    function emergencyRemoveUser(address user) public onlyAdmin {
        if (commuters[user].commuterId > 0) {
            uint256 commuterId = commuters[user].commuterId;
            delete commuters[user];
            delete commuterIdToAddress[commuterId];
        } else if (providers[user].providerId > 0) {
            uint256 providerId = providers[user].providerId;
            delete providers[user];
            delete providerIdToAddress[providerId];
        }
    }
    
    // ================ View Functions ================
    
    /**
     * @dev Get commuter information
     * @param commuter Commuter address
     */
    function getCommuter(address commuter) public view returns (
        uint256 commuterId,
        uint256 registrationTime,
        uint256 tripCount,
        uint256 totalSpent,
        uint256 reputationScore,
        bool isActive
    ) {
        Commuter storage c = commuters[commuter];
        return (
            c.commuterId,
            c.registrationTime,
            c.tripCount,
            c.totalSpent,
            c.reputationScore,
            c.isActive
        );
    }
    
    /**
     * @dev Get provider information
     * @param provider Provider address
     * @return Provider data
     */
    function getProvider(address provider) public view returns (Provider memory) {
        return providers[provider];
    }
    
    /**
     * @dev Get active providers by mode
     * @param mode Transport mode
     * @return Array of provider IDs
     */
    function getActiveProvidersByMode(TransportMode mode) public view returns (uint256[] memory) {
        // Count matching providers
        uint256 count = 0;
        for (uint256 i = 1; i <= numberOfProviders; i++) {
            address providerAddr = providerIdToAddress[i];
            if (providerAddr != address(0) && providers[providerAddr].modeType == mode && providers[providerAddr].isActive) {
                count++;
            }
        }
        
        // Populate array
        uint256[] memory result = new uint256[](count);
        uint256 index = 0;
        for (uint256 i = 1; i <= numberOfProviders; i++) {
            address providerAddr = providerIdToAddress[i];
            if (providerAddr != address(0) && providers[providerAddr].modeType == mode && providers[providerAddr].isActive) {
                result[index] = i;
                index++;
            }
        }
        
        return result;
    }
    
    /**
     * @dev Get feedback about a subject
     * @param subject Subject address
     * @param startIndex Start index for pagination
     * @param count Number of feedback items to return
     * @return Array of feedback
     */
    function getFeedback(address subject, uint256 startIndex, uint256 count) public view returns (Feedback[] memory) {
        require(startIndex < feedbackCount[subject], "Start index out of bounds");
        
        uint256 itemsToReturn = feedbackCount[subject] - startIndex;
        if (count < itemsToReturn) {
            itemsToReturn = count;
        }
        
        Feedback[] memory result = new Feedback[](itemsToReturn);
        uint256 resultIndex = 0;
        
        // Iterate through possible submitters to find feedback
        for (uint256 i = 1; i <= numberOfCommuters + numberOfProviders; i++) {
            address submitter;
            if (i <= numberOfCommuters) {
                submitter = commuterIdToAddress[i];
            } else {
                submitter = providerIdToAddress[i - numberOfCommuters];
            }
            
            if (submitter == address(0)) continue;
            
            Feedback[] storage feedbacks = feedbackHistory[subject][submitter];
            for (uint256 j = 0; j < feedbacks.length; j++) {
                if (resultIndex >= itemsToReturn) {
                    return result;
                }
                
                if (feedbacks[j].feedbackId >= startIndex) {
                    result[resultIndex] = feedbacks[j];
                    resultIndex++;
                }
            }
        }
        
        return result;
    }
    
    /**
     * @dev Get verified status of a user
     * @param user User address
     * @return Verification status
     */
    function isUserVerified(address user) public view returns (bool) {
        return publicProfiles[user].isVerified;
    }

    // Add these functions to MaaSRegistry.sol to expose needed data
    function getProviderReputation(address providerAddress) public view returns (uint256) {
        return reputationScores[providerAddress];
    }

    function isProviderVerified(address providerAddress) public view returns (bool) {
        // Simple verification: reputation > 80 or can add separate mapping
        return reputationScores[providerAddress] >= 80;
    }
}