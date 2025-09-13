// abm-interface.js
// Minimal JS SDK to talk to the simplified contracts via MaaSFacade
// Works with Ethers v6

import { ethers } from "ethers";
import fs from "fs";
import path from "path";

// ---------- Load deployment addresses ----------
const ADDR_PATH = path.resolve("./deployed/simplified.json");
if (!fs.existsSync(ADDR_PATH)) {
  throw new Error("Missing deployed/simplified.json. Deploy first.");
}
const ADDR = JSON.parse(fs.readFileSync(ADDR_PATH, "utf-8"));

// ---------- ABIs (only Facade ABI is strictly required for writes) ----------
// If you prefer, you can import build artifacts instead.
const FACADE_ABI = [
  "function owner() view returns (address)",
  "function marketplaceAPI() view returns (address)",

  "function setMarketplaceAPI(address api) external",
  "function wireMarketplace() external",

  "function registerAsCommuter(uint256 commuterId, address account) external",
  "function registerAsProvider(uint256 providerId, address account, uint8 mode) external",

  "function submitRequestHash(uint256 commuterId, string contentHash) external returns (uint256 requestId)",
  "function setRequestStatus(uint256 requestId, uint8 status) external",
  "function submitOfferHash(uint256 requestId, uint256 providerId, string contentHash) external returns (uint256 offerId)",
  "function recordMatch(uint256 requestId, uint256 offerId, uint256 providerId, uint256 priceWei) external",
  "function confirmCompletion(uint256 requestId) external",

  // Optional views from underlying contracts if you want to check state:
  // We'll connect directly to Request/Auction when needed below.
];

const REQUEST_ABI = [
  "function getRequestHash(uint256 requestId) view returns (string)",
  "function lastRequestId() view returns (uint256)"
];

const AUCTION_ABI = [
  "function getMatchResult(uint256 requestId) view returns (tuple(bool exists,uint256 requestId,uint256 offerId,uint256 providerId,uint256 priceWei,bool completed))",
  "function getOffers(uint256 requestId) view returns (tuple(uint256 providerId,string contentHash,address submittedBy)[])"
];

// ---------- Provider & signer ----------
/**
 * Create a signer + contract handles.
 * @param {Object} opts
 * @param {string} opts.rpc - RPC url (e.g., http://127.0.0.1:8545)
 * @param {string} opts.privateKey - EOA used as marketplaceAPI
 */
export function makeClients({ rpc, privateKey }) {
  if (!rpc || !privateKey) throw new Error("rpc and privateKey are required");

  const provider = new ethers.JsonRpcProvider(rpc);
  const signer = new ethers.Wallet(privateKey, provider);

  const facade = new ethers.Contract(ADDR.Facade, FACADE_ABI, signer);
  const request = new ethers.Contract(ADDR.Request, REQUEST_ABI, provider); // read-only ok
  const auction = new ethers.Contract(ADDR.Auction, AUCTION_ABI, provider); // read-only ok

  return { provider, signer, facade, request, auction, ADDR };
}

// ---------- Admin wiring (one-off after deploy) ----------
export async function wireMarketplaceAPI({ facade, signer }) {
  const currApi = await facade.marketplaceAPI();
  if (currApi.toLowerCase() !== (await signer.getAddress()).toLowerCase()) {
    const tx = await facade.setMarketplaceAPI(await signer.getAddress());
    await tx.wait();
  }
  // wire Request/Auction to accept the Facade as marketplace
  const tx2 = await facade.wireMarketplace();
  await tx2.wait();
  return true;
}

// ---------- Identity registration (owner-only on Facade) ----------
export async function registerCommuter({ facade }, commuterId, account) {
  const tx = await facade.registerAsCommuter(commuterId, account);
  const rc = await tx.wait();
  return rc.hash;
}

export async function registerProvider({ facade }, providerId, account, mode /* uint8 */) {
  const tx = await facade.registerAsProvider(providerId, account, mode);
  const rc = await tx.wait();
  return rc.hash;
}

// ---------- Request lifecycle (API-only) ----------
/**
 * Create a request referencing off-chain content (IPFS/URI)
 * @returns {Promise<number>} requestId
 */
export async function submitRequestHash({ facade }, commuterId, contentHash) {
  const tx = await facade.submitRequestHash(commuterId, contentHash);
  const rc = await tx.wait();
  // Ethers v6 returns logs; decode the return value via interface or read lastRequestId
  // Easiest: read the counter from Request
  // If you want the exact id from logs, wire an event & decode here.
  // Here we’ll just fetch lastRequestId from Request contract:
  // (We need a request client: pass request or reconstruct)
  // For robustness, expose request in the caller and read it there.
  return true;
}

/**
 * Convenience: submit + return the new requestId by reading Request.lastRequestId afterwards.
 */
export async function submitRequestHashAndGetId({ facade, request }, commuterId, contentHash) {
  const tx = await facade.submitRequestHash(commuterId, contentHash);
  await tx.wait();
  const id = await request.lastRequestId();
  return Number(id);
}

/**
 * Set request status: 0=None,1=Created,2=Matched,3=Completed,4=Cancelled
 */
export async function setRequestStatus({ facade }, requestId, statusEnum) {
  const tx = await facade.setRequestStatus(requestId, statusEnum);
  const rc = await tx.wait();
  return rc.hash;
}

// ---------- Offers & matching (API-only) ----------
/**
 * Submit an offer hash (on behalf of provider)
 * @returns {Promise<number>} offerId
 */
export async function submitOfferHash({ facade, auction }, requestId, providerId, contentHash) {
  const tx = await facade.submitOfferHash(requestId, providerId, contentHash);
  await tx.wait();

  // Offer ID is index in offersByRequest; safest is to read offers length-1
  const offers = await auction.getOffers(requestId);
  return offers.length - 1;
}

/**
 * Record the chosen match and (optionally) set request to Matched in Facade.
 * @param {bigint|string|number} priceWei - pass as BigInt or parse via ethers.parseEther("10")
 */
export async function recordMatch({ facade }, requestId, offerId, providerId, priceWei) {
  // normalize price
  const price = typeof priceWei === "bigint" ? priceWei : ethers.toBigInt(priceWei);
  const tx = await facade.recordMatch(requestId, offerId, providerId, price);
  const rc = await tx.wait();
  return rc.hash;
}

export async function confirmCompletion({ facade }, requestId) {
  const tx = await facade.confirmCompletion(requestId);
  const rc = await tx.wait();
  return rc.hash;
}

// ---------- Read helpers ----------
export async function getRequestHash({ request }, requestId) {
  return request.getRequestHash(requestId);
}

export async function getMatchResult({ auction }, requestId) {
  const res = await auction.getMatchResult(requestId);
  // shape: [exists, requestId, offerId, providerId, priceWei, completed]
  return {
    exists: res.exists,
    requestId: Number(res.requestId),
    offerId: Number(res.offerId),
    providerId: Number(res.providerId),
    priceWei: res.priceWei.toString(),
    completed: res.completed
  };
}
