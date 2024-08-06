from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader

from lit_gpt.datamodules.deita import Deita
from lit_gpt.utils import CycleIterator
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.datamodules.base import get_sft_collate_fn

tokenizer = Tokenizer(Path('tokenizer/pints'))


def test_deita_e2e():
    batch_size = 2

    deita = Deita(prompt_style='chatml')
    deita.tokenizer = tokenizer
    deita.include_multiturn_conversations = True

    # overwrite the prepare_data method to load custom
    def mock_prepare_data():
        # We do it like this to make it return a (train, test) tuple that Deita has.
        mock_train = load_dataset(
            'json',
            # Even though jsonl file doesn't have a split, `train` is needed by default
            split=['train'],
            data_files='lit_gpt/datamodules/deita_test.jsonl',
        )

        mock_test = load_dataset(
            'json',
            # Even though jsonl file doesn't have a split, `train` is needed by default
            split=['train'],
            data_files='lit_gpt/datamodules/deita_test.jsonl',
        )
        return (mock_train[0], mock_test[0])

    deita.prepare_data = mock_prepare_data

    data = deita.setup()

    train_dataloader = DataLoader(
        data['train_dataset'],
        batch_size=batch_size,
        collate_fn=get_sft_collate_fn(2048),
    )

    deita_iterator = CycleIterator(train_dataloader)

    iter_batch = next(deita_iterator)

    # Check that we plucked out according to batch_size
    assert len(iter_batch['input_ids']) == batch_size
    assert len(iter_batch['labels']) == batch_size

    decoded_input = tokenizer.decode(
        iter_batch['input_ids'][1], skip_special_tokens=False
    )

    expected_input = """<s><|im_start|> user
Hello<|im_end|> 
<|im_start|> assistant
Hello! How can I assist you today?<|im_end|> 
<|im_start|> user
I require your assistance to rewrite a service page about Disputing Extension of Time Claims for the persona of a Homeowner in the context of residential construction in {Adelaide, AU}}. The reference article is written for New South Wales. You will need to apply laws and legislations relevant to {Adelaide, AU} for the rewritten article. For now, just acknowledge my message.<|im_end|> 
<|im_start|> assistant
Thank you for your message. I understand that you need assistance in rewriting a service page about disputing extension of time claims for a homeowner in the context of residential construction in Adelaide, AU, while applying laws and legislations relevant to Adelaide. I acknowledge your request and I am ready to assist you with this task. Please provide me with any additional information or requirements you have for the rewritten article.<|im_end|> 
<|im_start|> user
I want you to strictly respond only in Australian English language. I want you to act as a SEO professional and legal content writing expert. Do you understand? Please write in AU English language.<|im_end|> 
<|im_start|> assistant
G'day! No worries, mate. I understand your requirements and I'll make sure to strictly use Australian English in my responses. As a language model, I can definitely assist you in creating a well-optimized and legally compliant content. Just let me know how I can be of assistance in rewriting the service page on disputing extension of time claims for a homeowner in the context of residential construction in Adelaide, AU. Cheers!<|im_end|> 
<|im_start|> user
You are to assume that you are a construction lawyer with more than 10 years specialist experience in this area of law. You want to rewrite a service page that is informative and helpful to the target persona, and considering the relevant laws and legislations applicable in {Adelaide, Australia}. Do you understand? Please write in AU English language<|im_end|> 
<|im_start|> assistant
Sure, no worries! As a construction lawyer with over a decade of experience in this area of law, I understand that the target audience for this service page is homeowners in Adelaide, Australia who are looking for information and assistance in disputing extension of time claims in the context of residential construction. My aim is to provide informative and helpful content while also taking into consideration the relevant laws and regulations applicable in Adelaide. Please let me know if you have any specific requirements or guidelines for the rewritten service page. Cheers!<|im_end|> 
<|im_start|> user
I want you to incorporate the following target keywords and phrases in my service page. It should be injected in the most natural way. Do you acknowledge?
extension of time disputes, extension of time claim, construction lawyers in adelaide<|im_end|> 
<|im_start|> assistant
Thank you for providing the target keywords and phrases for the rewritten service page. I acknowledge your request and will ensure that these keywords and phrases are incorporated in a natural and organic way to maximize SEO value while still providing valuable information to the target audience. Please let me know if you have any additional requirements or guidelines for the content. Cheers!<|im_end|> 
<|im_start|> user
I will send you the reference article per part. Please rewrite also by parts. Do you understand?<|im_end|> 
<|im_start|> assistant
Understood! Please feel free to send me the reference article and I will rewrite it per part as per your instructions. Let me know if you have any further questions or requirements. Cheers!<|im_end|> 
<|im_start|> user
Part 1: 

Dealing with construction delays as a homeowner can be frustrating and costly. If a builder claims an Extension of Time, it can add significant time and expense to the project. Understand your rights and obligations under the HIA Contract and take action if the claim is unjustified. Our experienced construction lawyers can assist you in disputing the claim and achieving the best possible outcome for your project, no matter the scale of the delay.

Ensure to apply the relevant laws and legislation in {Adelaide, Australia} during the rewriting process and target to Home Owners as the persona. Replace all information, laws, and governing bodies relevant to Adelaide or {Adelaide, Australia} with information, laws, and governing bodies relevant only to {Adelaide, Australia}.

Change Contracts Specialist to (Construction Lawyer Adelaide

Please rewrite the above considering the information given and providing unique content that is applicable for the target location, {Adelaide, Australia}.
Do not change the POV.<|im_end|> 
<|im_start|> assistant
Sure, I understand your requirements. Here's the rewritten Part 1:

As a homeowner in Adelaide, Australia, dealing with construction delays can be a frustrating and costly experience. When a builder claims an extension of time, it can significantly prolong the project and increase expenses. However, it's important to know your rights and obligations under the relevant laws and regulations in Adelaide, including the HIA Contract. If you believe that the extension of time claim is unjustified, you can take action to dispute it and achieve the best possible outcome for your project.

As construction lawyers in Adelaide with years of experience, we can provide you with the necessary legal expertise and assistance in disputing the extension of time claim. Whether you're dealing with a small-scale or large-scale delay, we can help you understand the laws and regulations applicable to your situation, and work towards a favorable resolution. So, if you're facing an extension of time dispute, don't hesitate to contact us - we're here to help.

Please note that we've replaced "Contracts Specialist" with "Construction Lawyer Adelaide" to better reflect our expertise and services.<|im_end|> 
<|im_start|> user
Part 2: 

Understanding Your Rights as a Homeowner
As a homeowner, it is important to understand your rights and obligations under the HIA Contract when dealing with construction delays and extensions of time. The following information will help you navigate the process of disputing an Extension of Time Claim:

Review the HIA Contract: The HIA Contract sets out the terms and conditions of the construction project, including the builder’s entitlement to an extension of time. Reviewing this contract will help you understand your rights and obligations.
Provide written notice of dispute: If you disagree with the builder’s claim for an extension of time, you must provide written notice of dispute within 5 working days of receiving the builder’s notice. This notice should detail the reasons why you are disputing the Extension of Time Claim.
Negotiate in good faith: The HIA Contract requires both parties to negotiate in good faith and make reasonable efforts to resolve any disputes. This means that you should communicate clearly and openly with the builder and work towards a mutually beneficial solution.
Seek legal advice: If you are unsure of your rights or need assistance in disputing an Extension of Time Claim, seek legal advice from an experienced construction lawyer.
Understanding your rights as a homeowner is the first step in disputing an Extension of Time Claim. By familiarising yourself with the terms of the HIA Contract and taking action in a timely manner, you can protect your interests and ensure that your construction project stays on track.

Ensure to apply the relevant laws and legislation in {Adelaide, Australia} during the rewriting process and target to Home Owners as the persona. Replace all information, laws, and governing bodies relevant to Adelaide or {Adelaide, Australia} with information, laws, and governing bodies relevant only to {Adelaide, Australia}.

Change Contracts Specialist to Construction Lawyer Adelaide

Please rewrite the above considering the information given and providing unique content that is applicable for the target location, {Adelaide, Australia}.
Do not change the POV.<|im_end|> 
<|im_start|> assistant
Certainly! Here's the rewritten Part 2:

Understanding your rights as a homeowner in Adelaide, Australia is crucial when dealing with construction delays and extensions of time. If you receive a claim for an extension of time from your builder, it's important to take the following steps:

Review the HIA Contract: The HIA Contract outlines the terms and conditions of your construction project, including the builder's entitlement to an extension of time. Understanding this contract will help you better comprehend your rights and obligations.
Provide written notice of dispute: If you disagree with the builder's claim for an extension of time, you must provide written notice of dispute within 5 working days of receiving the builder's notice. This written notice should clearly outline the reasons why you are disputing the Extension of Time Claim.
Negotiate in good faith: The HIA Contract requires both parties to negotiate in good faith and make reasonable efforts to resolve any disputes. It's important to communicate clearly and openly with the builder and work towards a mutually beneficial solution.
Seek legal advice: If you're unsure about your rights or need help in disputing an Extension of Time Claim, consult an experienced construction lawyer in Adelaide.

By familiarising yourself with the terms of the HIA Contract and taking timely action, you can protect your interests and keep your construction project on track. At Construction Lawyer Adelaide, we can provide you with the legal expertise you need to navigate this complex process. So, don't hesitate to seek our assistance if you're dealing with an<|im_end|>"""

    labels = iter_batch['labels'][1]
    labels[labels == -100] = 0
    decoded_labels = tokenizer.decode(
        iter_batch['labels'][1], skip_special_tokens=False
    )

    expected_labels = """<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>Hello! How can I assist you today?<|im_end|><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>Thank you for your message. I understand that you need assistance in rewriting a service page about disputing extension of time claims for a homeowner in the context of residential construction in Adelaide, AU, while applying laws and legislations relevant to Adelaide. I acknowledge your request and I am ready to assist you with this task. Please provide me with any additional information or requirements you have for the rewritten article.<|im_end|><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>G'day! No worries, mate. I understand your requirements and I'll make sure to strictly use Australian English in my responses. As a language model, I can definitely assist you in creating a well-optimized and legally compliant content. Just let me know how I can be of assistance in rewriting the service page on disputing extension of time claims for a homeowner in the context of residential construction in Adelaide, AU. Cheers!<|im_end|><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>Sure, no worries! As a construction lawyer with over a decade of experience in this area of law, I understand that the target audience for this service page is homeowners in Adelaide, Australia who are looking for information and assistance in disputing extension of time claims in the context of residential construction. My aim is to provide informative and helpful content while also taking into consideration the relevant laws and regulations applicable in Adelaide. Please let me know if you have any specific requirements or guidelines for the rewritten service page. Cheers!<|im_end|><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>Thank you for providing the target keywords and phrases for the rewritten service page. I acknowledge your request and will ensure that these keywords and phrases are incorporated in a natural and organic way to maximize SEO value while still providing valuable information to the target audience. Please let me know if you have any additional requirements or guidelines for the content. Cheers!<|im_end|><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>Understood! Please feel free to send me the reference article and I will rewrite it per part as per your instructions. Let me know if you have any further questions or requirements. Cheers!<|im_end|><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>Sure, I understand your requirements. Here's the rewritten Part 1:

As a homeowner in Adelaide, Australia, dealing with construction delays can be a frustrating and costly experience. When a builder claims an extension of time, it can significantly prolong the project and increase expenses. However, it's important to know your rights and obligations under the relevant laws and regulations in Adelaide, including the HIA Contract. If you believe that the extension of time claim is unjustified, you can take action to dispute it and achieve the best possible outcome for your project.

As construction lawyers in Adelaide with years of experience, we can provide you with the necessary legal expertise and assistance in disputing the extension of time claim. Whether you're dealing with a small-scale or large-scale delay, we can help you understand the laws and regulations applicable to your situation, and work towards a favorable resolution. So, if you're facing an extension of time dispute, don't hesitate to contact us - we're here to help.

Please note that we've replaced "Contracts Specialist" with "Construction Lawyer Adelaide" to better reflect our expertise and services.<|im_end|><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>Certainly! Here's the rewritten Part 2:

Understanding your rights as a homeowner in Adelaide, Australia is crucial when dealing with construction delays and extensions of time. If you receive a claim for an extension of time from your builder, it's important to take the following steps:

Review the HIA Contract: The HIA Contract outlines the terms and conditions of your construction project, including the builder's entitlement to an extension of time. Understanding this contract will help you better comprehend your rights and obligations.
Provide written notice of dispute: If you disagree with the builder's claim for an extension of time, you must provide written notice of dispute within 5 working days of receiving the builder's notice. This written notice should clearly outline the reasons why you are disputing the Extension of Time Claim.
Negotiate in good faith: The HIA Contract requires both parties to negotiate in good faith and make reasonable efforts to resolve any disputes. It's important to communicate clearly and openly with the builder and work towards a mutually beneficial solution.
Seek legal advice: If you're unsure about your rights or need help in disputing an Extension of Time Claim, consult an experienced construction lawyer in Adelaide.

By familiarising yourself with the terms of the HIA Contract and taking timely action, you can protect your interests and keep your construction project on track. At Construction Lawyer Adelaide, we can provide you with the legal expertise you need to navigate this complex process. So, don't hesitate to seek our assistance if you're dealing with an<|im_end|>"""

    assert expected_input == decoded_input
    assert expected_labels == decoded_labels
