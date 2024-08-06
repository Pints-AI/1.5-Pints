from json import loads
from lit_gpt.datamodules.llama_instruct import format_dataset

MOCK_DATA = []

with open(
    './lit_gpt/datamodules/llama_instruct_test.jsonl', 'r', encoding='utf-8'
) as jsonl_file:
    for line in jsonl_file:
        MOCK_DATA.append(loads(line))


def test_format_dataset_multiturn():
    train_data = format_dataset(
        dataset=MOCK_DATA, include_multi_turn_conversations=True
    )

    train_data_row_one = train_data[0]
    # Hard code cause if we do the same string manipulation process, it's just not testing the code.
    mock_data_row_one = {
        'instruction': '''root@openvpn:/home/openvpn# ./openvpn-install.sh\nWelcome to OpenVPN-install!\nThe git repository is available at: https://github.com/angristan/openvpn-install\n\nIt looks like OpenVPN is already installed.\n\nWhat do you want to do?\n   1) Add a new user\n   2) Revoke existing user\n   3) Remove OpenVPN\n   4) Exit\nSelect an option [1-4]: 1\n\nTell me a name for the client.\nThe name must consist of alphanumeric character. It may also include an underscore or a dash.\nClient name: naam\n\nDo you want to protect the configuration file with a password?\n(e.g. encrypt the private key with a password)\n   1) Add a passwordless client\n   2) Use a password for the client\nSelect an option [1-2]: 1\n\nNote: using Easy-RSA configuration from: /etc/openvpn/easy-rsa/vars\nUsing SSL: openssl OpenSSL 3.0.2 15 Mar 2022 (Library: OpenSSL 3.0.2 15 Mar 2022)\n-----\nUsing configuration from /etc/openvpn/easy-rsa/pki/easy-rsa-54848.BT2FXv/tmp.dFLd6V\nEnter pass phrase for /etc/openvpn/easy-rsa/pki/private/ca.key:\nCheck that the request matches the signature\nSignature ok\nThe Subject's Distinguished Name is as follows\ncommonName            :ASN.1 12:'naam'\nCertificate is to be certified until Apr 21 22:48:57 2025 GMT (825 days)\n\nWrite out database with 1 new entries\nData Base Updated\n\nClient naam added.\n\nThe configuration file has been written to /home/openvpn/naam.ovpn.\nDownload the .ovpn file and import it in your OpenVPN client.\nroot@openvpn:/home/openvpn# \n\n\nkan je op basis van deze gegevens een verslag maken hoe je iemand toevoegd op een openvpn server''',
        'output': '''OpenVPN-installer shell script has been used to add a new user to an existing OpenVPN server. The script can be found on GitHub.\n\nThe user has been prompted to enter a name for the client, which must consist of alphanumeric characters and may include an underscore or a dash. The name \"naam\" has been entered.\n\nThe user has been asked whether to protect the configuration file with a password. The user has chosen not to add a password for the client.\n\nThe script has used the Easy-RSA configuration from /etc/openvpn/easy-rsa/vars and has generated a private key for the client using OpenSSL. The user has been prompted to enter a pass phrase for the private key.\n\nThe script has then generated a certificate for the client using the entered name and the private key. The certificate is valid until April 21, 2025.\n\nThe script has written the new entry to the database and has updated the configuration file. The client's configuration file has been written to /home/openvpn/naam.ovpn, and the user has been instructed to download the file and import it into their OpenVPN client.\n\nIn conclusion, the OpenVPN-installer script has been used to successfully add a new user to an existing OpenVPN server. The user has been prompted to enter a name for the client and has been given the option to protect the configuration file with a password. The script has generated a private key and certificate for the client and has updated the configuration file. The user can now download the configuration file and import it into their OpenVPN client.'''
    }

    # The instruction and output pair in train_data is made
    # from `content` from the nth and nth+1 row in `messages``
    assert train_data_row_one[0]['instruction'] == mock_data_row_one['instruction']
    assert train_data_row_one[0]['output'] == mock_data_row_one['output']
